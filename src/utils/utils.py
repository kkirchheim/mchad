import logging
import warnings
from typing import List, Sequence, Any

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from pytorch_lightning.utilities import rank_zero_only
import torch
import numpy as np
import time


from pytorch_ood.utils import is_known, is_unknown, contains_unknown, contains_known
from tensorboardX import SummaryWriter

log = logging.getLogger(__name__)


def outputs_detach_cpu(d):
    """
    Detaches all tensors in the given dict and sends them to gpu, except for the tensor with the key
    "loss"
    """
    new_d = {k: v.detach().cpu() for k, v in d.items() if k != "loss"}
    if "loss" in d:
        new_d["loss"] = d["loss"]
    return new_d


def load_pretrained_checkpoint(model, pretrained_checkpoint):
    log.info(f"Loading pretrained weights from {pretrained_checkpoint}")
    state_dict = torch.load(pretrained_checkpoint, map_location=torch.device("cpu"))
    del state_dict["module.fc.weight"]
    del state_dict["module.fc.bias"]
    model.load_state_dict(state_dict, strict=False)


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")
        config.trainer.fast_dev_run = True

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(
            f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>"
        )
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "testmodules",
        "callbacks",
        "logger",
        "seed",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""
    pass


###################################################
# Helpers for logging to tensorboard/loggers
###################################################


def get_tensorboard(obj) -> SummaryWriter:
    """
    Helper function to get the tensorboard for a module from a list of its loggers
    """
    if isinstance(obj, pl.LightningModule):
        loggers = obj.logger
    elif isinstance(obj, LoggerCollection):
        loggers = obj
    else:
        raise TypeError(f"Unknown Type: {type(obj)}")

    if loggers is None:
        return None

    if isinstance(loggers, TensorBoardLogger):
        return loggers.experiment

    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger.experiment


def _get_tb(loggers) -> TensorBoardLogger:
    """
    Gets the tensorboard logger from model.logger
    """
    if isinstance(loggers, TensorBoardLogger):
        return loggers

    for logger in loggers:
        if isinstance(logger, TensorBoardLogger):
            return logger


def log_weight_hists(model: pl.LightningModule):
    if model.logger is None:
        return

    for name, param in model.named_parameters():
        get_tensorboard(model.logger).add_histogram(
            tag=f"weights/{name}", values=param, global_step=model.global_step
        )


def log_grad_hists(model: pl.LightningModule):
    if model.logger is None:
        return

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            get_tensorboard(model.logger).add_histogram(
                tag=f"gradients/{name}",
                values=param.grad,
                global_step=model.global_step,
            )


def log_score_histogram(model, stage, score, y, y_hat, method=None):
    """
    Save histograms of confidence scores
    """

    writer = get_tensorboard(model)
    epoch = model.current_epoch

    if writer:
        correct = y == y_hat
        incorrect = ~correct

        known = is_known(y)
        unknown = is_unknown(y)

        if method:
            prefix = f"Score/{stage}/{method}"
        else:
            prefix = f"Score/{stage}"

        if contains_known(y):
            if (known & correct).any() and not np.isnan(score).any():
                writer.add_histogram(
                    tag=f"{prefix}/known/correct",
                    values=score[known & correct],
                    global_step=epoch,
                )

            if (known & incorrect).any() and not np.isnan(score).any():
                writer.add_histogram(
                    tag=f"{prefix}/known/incorrect",
                    values=score[known & incorrect],
                    global_step=epoch,
                )

        if contains_unknown(y) and not np.isnan(score).any():
            writer.add_histogram(
                tag=f"{prefix}/unknown", values=score[unknown], global_step=epoch
            )


class ContextGuard:
    def __init__(self, experiment, name=None):
        self.experiment = experiment
        self.name = name
        self.t0 = None
        self.t1 = None

    def __repr__(self):
        return f"ContextGuard({self.name})"

    def __enter__(self):
        self.t0 = time.time()

        if self.name:
            log.debug(f"Entering Context: '{self.name}'")

    def __exit__(self, exc_type, exc_val, exc_traceback):
        self.t1 = time.time()

        if self.name:
            log.debug(f"Context '{self.name}' left after {self.t1 - self.t0} s.")

        if exc_type is KeyboardInterrupt:
            log.warning("Keyboard Interrupt.")
            return False

        if exc_type:
            log.error(f"Exception in '{self.experiment}' in context '{self.name}'")
            log.exception(exc_type)
            log.exception(exc_val)
            log.exception(exc_traceback)
            return False

        return True


def collect_outputs(outputs: List[Any], key) -> torch.Tensor:
    """
    Collect outputs for model with multiple data loaders, which will be a list of dicts.

    :param outputs: outputs returned by pytorch lighting.
    :param key: the key to collect all tensors for, e.g. "embeddings"
    """
    if type(outputs) is list:
        # multiple data loaders
        # i have no idea when which case hits ...
        if type(outputs[0]) is list:
            if type(outputs[0][0]) is dict:
                tmp = []
                for output in outputs:
                    tmp.extend([o[key] for o in output])
            else:
                tmp = []
                for output in outputs:
                    tmp.extend([o for o in output])
            return torch.cat(tmp)
        elif type(outputs[0]) is dict:
            return torch.cat([output[key] for output in outputs])
        else:
            tmp = []
            for output in outputs:
                tmp.extend([o for o in output])
            return torch.cat(tmp)
    else:
        return torch.cat([output[key] for output in outputs])


def save_embeddings(
    pl_model: pl.LightningModule,
    dists=None,
    embedding=None,
    images=None,
    targets=None,
    tag="default",
    limit=10000,
):
    """
    Helper for saving embeddings etc. to tensorboard

    :param pl_model: pytorch lightning module
    :param dists: distances of samples to some reference points, for example class centers. Only minimum will be logged.
    :param embedding: representation of input points
    :param images: corresponding input points. Can be none
    :param targets: labels
    :param tag: tag to log to tensorboard
    :param limit: maximum number of instances to log. to many will result in the generated files being to large etc.
    """

    log.info("Saving embeddings")

    # limit number of saved entries so tensorboard does not crash because of too many sprites
    if limit:
        indexes = torch.randperm(len(embedding))[:limit]
    else:
        indexes = torch.arange(len(embedding))

    args = {
        "known": is_known(targets)[indexes],
        "labels": targets[indexes],
    }

    if dists is not None:
        args.update({"distance": torch.min(dists[indexes], dim=1)[0]})
        args.update({"predictions": torch.argmin(dists[indexes], dim=1)})

    header, data = create_metadata(**args)

    get_tensorboard(pl_model).add_embedding(
        embedding[indexes],
        metadata=data,
        global_step=pl_model.global_step,
        metadata_header=header,
        label_img=None if images is None else images[indexes],
        tag=tag,
    )


def create_metadata(**kwargs):
    """
    Create metadata to save embeddings with tensorboard

    :param kwargs: keys and an associated list of values
    """
    header = [k for k, _ in kwargs.items()]
    data = [[str(c.item()) for c in col] for col in zip(*kwargs.values())]

    return header, data
