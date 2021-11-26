#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common functions accessed by other modules
"""
import json
import logging
import sys
import time
import types
from collections import defaultdict
from os.path import join
from typing import List, Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as scheduler
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import LoggerCollection
from torch.utils.data import Subset

from osr.utils import is_known, is_unknown, contains_known, contains_unknown

log = logging.getLogger(__name__)


def configure_logging(path=None, stderr=False):
    fmt = "[%(levelname)s] (%(processName)s)  %(asctime)s - %(name)s: %(message)s"

    if path and path.strip() != "-":
        logging.basicConfig(filename=path, level=logging.DEBUG, format=fmt)

    root = logging.getLogger()

    if path == "-":
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(fmt=fmt))
        root.addHandler(ch)

    if stderr:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(logging.Formatter(fmt=fmt))
        root.addHandler(ch)

    root.setLevel(logging.DEBUG)
    root.info("Logging configured")


####################################################
# Creating object from configuration
###################################################


def create_optimizer(config, parameter):
    """
    @param config:
    @param parameter: a model or a list of parameters to optimize
    """
    lr = config["learning_rate"]
    weight_decay = config["weight_decay"]
    name = config["name"]
    momentum = config["momentum"]

    if name == "adam":
        # FIXME: adam does not support momentum as parameter
        if isinstance(parameter, (list, types.GeneratorType)):
            opti = torch.optim.Adam(parameter, lr=lr, weight_decay=weight_decay)
        else:
            opti = torch.optim.Adam(
                parameter.parameters(), lr=lr, weight_decay=weight_decay
            )
    elif name == "sgd":
        if isinstance(parameter, (list, types.GeneratorType)):
            opti = torch.optim.SGD(
                parameter, lr=lr, weight_decay=weight_decay, momentum=momentum
            )
        else:
            opti = torch.optim.SGD(
                parameter.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
            )
    else:
        raise ValueError(f"Unknown Optimizer: {name}")

    return opti


def create_scheduler(config, optimizer):
    if config["name"] == "MultiStepLR":
        return scheduler.MultiStepLR(
            optimizer, milestones=config["milestones"], gamma=config["gamma"]
        )
    elif config["name"] == "ExponentialLR":
        return scheduler.ExponentialLR(optimizer, gamma=config["gamma"])
    elif config["name"] == "CosineAnnealingLR":
        return scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"])
    else:
        return ValueError


#######################################################
# Helpers for loading and saving
#######################################################


def save_pipeline(directory, pipeline, train):
    if train:
        path = join(directory, "pipeline-train.pt")
    else:
        path = join(directory, "pipeline-test.pt")
    torch.save(pipeline, path)


def load_pipeline(directory, train):
    if train:
        path = join(directory, "pipeline-train.pt")
    else:
        path = join(directory, "pipeline-test.pt")

    return torch.load(path)


def save_config(config, directory):
    OmegaConf.save(config, join(directory, "config.yaml"))


def load_config(directory):
    return OmegaConf.load(join(directory, "config.yaml"))


def load_split(directory):
    return torch.load(join(directory, "split.pt"))


def save_split(split, directory):
    torch.save(split, join(directory, "split.pt"))


def save_target_mapping(directory, mapping):
    torch.save(mapping, join(directory, "class-mapping.pt"))

    # additionally, save as json for convenience
    path = join(directory, "class-mapping.json")
    m = {int(k): int(v) for k, v in mapping.items()}
    s = json.dumps(m, sort_keys=True, indent=4)
    # log.debug(f"Target Mapping: {s}")
    with open(path, "w") as f:
        f.write(s)


def load_target_mapping(directory):
    return torch.load(join(directory, "class-mapping.pt"))


def get_dataset(loader):
    """
    Unwrap dataset
    """
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        return get_dataset(dataset)
    else:
        return dataset


###################################################
# Helpers for logging to tensorboard/loggers
###################################################


def find_tensorboard(obj):
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


def get_tb_logger(loggers) -> TensorBoardLogger:
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
        find_tensorboard(model.logger).add_histogram(
            tag=f"weights/{name}", values=param, global_step=model.global_step
        )


def log_grad_hists(model: pl.LightningModule):
    if model.logger is None:
        return

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            find_tensorboard(model.logger).add_histogram(
                tag=f"gradients/{name}",
                values=param.grad,
                global_step=model.global_step,
            )


def log_score_histogram(model, stage, score, y, y_hat, method=None):
    """
    Save histograms of confidence scores
    """

    writer = find_tensorboard(model)
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


def create_metadata(known, labels, distance=None, centers=None):
    """
    Create metadata for embedding logging
    """
    if distance is not None:
        header = ["label", "known", "distance"]
        data = [
            [str(l.item()), str(k.item()), str(d.item())]
            for k, l, d in zip(known, labels, distance)
        ]
    else:
        header = ["label", "known"]
        data = [[str(l.item()), str(k.item())] for k, l in zip(known, labels)]

    return header, data


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


class TensorBuffer:
    """
    Used to buffer some tensors
    """

    def __init__(self, device="cpu"):
        self._buffer = defaultdict(list)
        self.device = device

    def append(self, key, value: torch.Tensor):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Can not handle value type {type(value)}")

        # log.debug(f"Adding tensor with key {key} to buffer shape=({value.size()})")
        value = value.detach().to(self.device)

        self._buffer[key].append(value)
        return self

    def __contains__(self, elem):
        return elem in self._buffer

    def __getitem__(self, item):
        return self.get(item)

    def sample(self, key) -> torch.Tensor:
        index = torch.randint(0, len(self._buffer[key]), size=(1,))
        return self._buffer[key][index]

    def get(self, key) -> torch.Tensor:
        if key not in self._buffer:
            raise KeyError(key)

        v = torch.cat(self._buffer[key])
        # log.debug(f"Retrieving from buffer {key} with shape={v.size()}")
        return v

    def clear(self):
        log.debug("Clearing buffer")
        self._buffer.clear()
        return self

    def save(self, path):
        """Save buffer to disk"""
        d = {k: self.get(k).cpu() for k in self._buffer.keys()}
        log.debug(f"Saving tensor buffer to {path}")
        torch.save(d, path)
        return self


##################################################


def collect_outputs(outputs: List[Any], key) -> torch.Tensor:
    """
    Collect outputs for model with multiple dataloaders
    """
    if type(outputs) is list:
        # multiple data loaders
        # i have no idea when which case hits ...
        if type(outputs[0]) is list:
            if type(outputs[0][0]) is dict:
                l = []
                for output in outputs:
                    l.extend([o[key] for o in output])
            else:
                l = []
                for output in outputs:
                    l.extend([o for o in output])
            return torch.cat(l)
        elif type(outputs[0]) is dict:
            return torch.cat([output[key] for output in outputs])
        else:
            l = []
            for output in outputs:
                l.extend([o for o in output])
            return torch.cat(l)
    else:
        return torch.cat([output[key] for output in outputs])


def save_embeddings(
    pl_model,
    dists=None,
    embedding=None,
    images=None,
    targets=None,
    centers=None,
    tag="default",
    limit=5000,
):
    # limit number of saved entries so tensorboard does not crash because of too many sprites
    log.info(f"Saving embeddings")

    indexes = torch.randperm(len(embedding))[:limit]
    header, data = create_metadata(
        is_known(targets[indexes]),
        targets[indexes],
        distance=None if dists is None else torch.min(dists[indexes], dim=1)[0],
        centers=centers,
    )

    find_tensorboard(pl_model).add_embedding(
        embedding[indexes],
        metadata=data,
        global_step=pl_model.global_step,
        metadata_header=header,
        label_img=None if images is None else images[indexes],
        tag=tag,
    )
