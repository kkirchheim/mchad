#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Common functions accessed by other modules
"""
import json
import logging
import os
import sys
import time
import types
from os.path import join

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as scheduler
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import LoggerCollection
from torch.utils.data import Subset
from collections import defaultdict
from typing import List, Any

from src.osr.utils import is_known, is_unknown, contains_known, contains_unknown

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
            opti = torch.optim.Adam(parameter.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        if isinstance(parameter, (list, types.GeneratorType)):
            opti = torch.optim.SGD(parameter, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            opti = torch.optim.SGD(parameter.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unknown Optimizer: {name}")

    return opti


def create_scheduler(config, optimizer):
    if config["name"] == "MultiStepLR":
        return scheduler.MultiStepLR(optimizer, milestones=config["milestones"], gamma=config["gamma"])
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


def get_dumpfile(model: pl.LightningModule, stage: str):
    """
    Get the filename under which the results for a certain stage should be stored, like embeddings etc.
    We try to guess which dataset is currently being processed to include this into the filename. In order for this to
    work, the dataset has to have a "name" attribute, and "train" "test" or "val" have to be passed. Otherwise, we will
    use the given stage name.

    :param model:
    :param stage: name of the stage the model is currently in.
    :return:
    """
    if model.logger is not None:
        logger = get_tb_logger(model.logger)
        if stage == "train" or stage == "val":
            # use tensorboards directory structure
            root = join(logger.log_dir, f"{model.current_epoch:05d}")
            os.makedirs(root, exist_ok=True)
        else:
            root = logger.log_dir
    else:
        try:
            import ray.tune
            root = ray.tune.get_trial_dir()
            if root is None:
                log.warning("Logging to .")
                root = "."
        except Exception as e:
            log.exception(e)
            root = "."

    # TODO: this might lead to errors during testing on noise?
    if stage == "test":
        dataset = get_dataset(model.test_dataloader())
    elif stage == "val":
        dataset = get_dataset(model.val_dataloader())
    elif stage == "train":
        dataset = get_dataset(model.train_dataloader())
    else:
        dataset = None

    if hasattr(dataset, "name"):
        dataset_name = getattr(dataset, "name")
        filename = f"dump-{stage}-{dataset_name}.pt"
    else:
        filename = f"dump-{stage}.pt"

    return join(root, filename)


###################################################
# Helpers for logging to tensorboard/loggers
###################################################

def get_tb_writer(obj):
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
        get_tb_writer(model.logger).add_histogram(
            tag=f"weights/{name}",
            values=param,
            global_step=model.global_step
        )


def log_grad_hists(model: pl.LightningModule):
    if model.logger is None:
        return

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            get_tb_writer(model.logger).add_histogram(
                tag=f"gradients/{name}",
                values=param.grad,
                global_step=model.global_step
            )


def log_score_histogram(model, stage, score, y, y_hat, method=None):
    """
    Save histograms of confidence scores
    """

    writer = get_tb_writer(model)
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
                    global_step=epoch)

            if (known & incorrect).any() and not np.isnan(score).any():
                writer.add_histogram(
                    tag=f"{prefix}/known/incorrect",
                    values=score[known & incorrect],
                    global_step=epoch)

        if contains_unknown(y) and not np.isnan(score).any():
            writer.add_histogram(
                tag=f"{prefix}/unknown",
                values=score[unknown],
                global_step=epoch)

#
# def log_error_detection_metrics(model: pl.LightningModule, score, stage, y, y_hat, method=None, prog_bar=False):
#     """
#     Log error-dectection metrics, AUROC and AUPR
#
#     Error Detection Refers to the ability to discriminate correctly classified and misclassified images.
#     It is unrelated to OOD/OSR, as it does not consider samples from unknown classes.
#     """
#     if not contains_known(y):
#         log.warning("Passed data does not contain known samples. Can not calculate error Metrics.")
#
#         if not method:
#             model.log(f"AUROC/Error/{stage}", np.nan)
#             model.log(f"AUPR-IN/Error/{stage}", np.nan)
#             model.log(f"AUPR-OUT/Error/{stage}", np.nan)
#             model.log(f"MeanConf/Error/correct/{stage}", np.nan)
#             model.log(f"MeanConf/Error/incorrect/{stage}", np.nan)
#         else:
#             model.log(f"AUROC/Error/{stage}/{method}", np.nan)
#             model.log(f"AUPR-IN/Error/{stage}/{method}", np.nan)
#             model.log(f"AUPR-OUT/Error/{stage}/{method}", np.nan)
#             model.log(f"MeanConf/Error/correct/{stage}/{method}", np.nan)
#             model.log(f"MeanConf/Error/incorrect/{stage}/{method}", np.nan)
#         return
#
#     correct_class = (y == y_hat).long()
#     known = is_known(y)
#
#     try:
#         auroc = metrics.auroc(score[known], correct_class[known])
#
#         # AUPR IN
#         precision, recall, thresholds = metrics.precision_recall_curve(score[known], correct_class[known], pos_label=1)
#         aupr_in = metrics.auc(recall, precision)
#
#         # AUPR OUT
#         precision, recall, thresholds = metrics.precision_recall_curve(-score[known], 1 - correct_class[known], pos_label=1)
#         aupr_out = metrics.auc(recall, precision)
#
#         if not method:
#             model.log(f"AUROC/Error/{stage}", auroc, prog_bar=prog_bar)
#             model.log(f"AUPR-IN/Error/{stage}", aupr_in, prog_bar=prog_bar)
#             model.log(f"AUPR-OUT/Error/{stage}", aupr_out, prog_bar=prog_bar)
#             model.log(f"MeanConf/Error/correct/{stage}", score[known & (y == y_hat)].mean(), prog_bar=prog_bar)
#             model.log(f"MeanConf/Error/incorrect/{stage}", score[known & ~(y == y_hat)].mean(), prog_bar=prog_bar)
#         else:
#             model.log(f"AUROC/Error/{stage}/{method}", auroc, prog_bar=prog_bar)
#             model.log(f"AUPR-IN/Error/{stage}/{method}", aupr_in, prog_bar=prog_bar)
#             model.log(f"AUPR-OUT/Error/{stage}/{method}", aupr_out, prog_bar=prog_bar)
#             model.log(f"MeanConf/Error/correct/{stage}/{method}", score[known & (y == y_hat)].mean(), prog_bar=prog_bar)
#             model.log(f"MeanConf/Error/incorrect/{stage}/{method}", score[known & ~(y == y_hat)].mean(), prog_bar=prog_bar)
#
#     except Exception as e:
#         log.warning(e)
#
#
# def log_uncertainty_metrics(model, score, stage, y, y_hat, method=None, prog_bar=False):
#     """
#     Log uncertainty metrics, AUROC and AUPR
#
#     Uncertainty refers to the ability to discriminate correctly classified images from
#     unknown or misclassified images.
#     """
#     if not contains_known_and_unknown(y):
#         log.warning("Passed data does not contain known and unknown samples. Can not calculate uncertainty Metrics.")
#
#         if method:
#             model.log(f"AUROC/Uncertainty/{stage}/{method}", np.nan)
#             model.log(f"AUPR-IN/Uncertainty/{stage}/{method}", np.nan)
#             model.log(f"AUPR-OUT/Uncertainty/{stage}/{method}", np.nan)
#         else:
#             model.log(f"AUROC/Uncertainty/{stage}", np.nan)
#             model.log(f"AUPR-IN/Uncertainty/{stage}", np.nan)
#             model.log(f"AUPR-OUT/Uncertainty/{stage}", np.nan)
#
#     else:
#         try:
#             known_and_correct = (is_known(y) & (y == y_hat))
#             auroc = metrics.auroc(score, known_and_correct)
#
#             # AUPR IN
#             precision, recall, thresholds = metrics.precision_recall_curve(
#                 score, known_and_correct.long(), pos_label=1)
#             aupr_in = metrics.auc(recall, precision)
#
#             # AUPR OUT
#             precision, recall, thresholds = metrics.precision_recall_curve(
#                 -score, 1-known_and_correct.long(), pos_label=1)
#
#             aupr_out = metrics.auc(recall, precision)
#
#             if not method:
#                 model.log(f"AUROC/Uncertainty/{stage}", auroc, prog_bar=prog_bar)
#                 model.log(f"AUPR-IN/Uncertainty/{stage}", aupr_in, prog_bar=prog_bar)
#                 model.log(f"AUPR-OUT/Uncertainty/{stage}", aupr_out, prog_bar=prog_bar)
#             else:
#                 model.log(f"AUROC/Uncertainty/{stage}/{method}", auroc, prog_bar=prog_bar)
#                 model.log(f"AUPR-IN/Uncertainty/{stage}/{method}", aupr_in, prog_bar=prog_bar)
#                 model.log(f"AUPR-OUT/Uncertainty/{stage}/{method}", aupr_out, prog_bar=prog_bar)
#
#         except Exception as e:
#             log.warning(e)
#
#     known = is_known(y)
#
#     if contains_known(y):
#         if not method:
#             model.log(f"MeanConf/Uncertainty/known/{stage}", score[known].mean(), prog_bar=prog_bar)
#         else:
#             model.log(f"MeanConf/Uncertainty/known/{stage}/{method}", score[known].mean(), prog_bar=prog_bar)
#
#     if contains_unknown(y):
#         if not method:
#             model.log(f"MeanConf/Uncertainty/unknown/{stage}", score[~known].mean(), prog_bar=prog_bar)
#         else:
#             model.log(f"MeanConf/Uncertainty/unknown/{stage}/{method}", score[~known].mean(), prog_bar=prog_bar)
#
#
# def log_classification_metrics(model: pl.LightningModule, stage, y, y_hat, logits=None):
#     """
#
#     """
#     if contains_known(y):
#         known_idx = is_known(y)
#         acc = metrics.accuracy(y_hat[known_idx], y[known_idx], num_classes=model.num_classes)
#         model.log(f"Accuracy/{stage}", acc, prog_bar=True)
#
#         if logits is not None:
#             loss = model.loss(logits[known_idx], y[known_idx])
#             model.log(f"Loss/{stage}", loss, prog_bar=True)
#     else:
#         log.error("Passed data does not contain known and unknown samples. Can not calculate OSR Metrics.")
#
#         model.log(f"Accuracy/{stage}", np.nan)
#
#         if logits is not None:
#             model.log(f"Loss/{stage}", np.nan)
#         return
#
#
# def log_osr_metrics(model: pl.LightningModule, score, stage, y, method=None, prog_bar=False):
#     """
#     Log uncertainty metrics, AUROC and AUPR
#
#     Uncertainty refers to the ability to discriminate images of known from images of unknown classes.
#     """
#
#     if not contains_known_and_unknown(y):
#         log.error("Passed data does not contain known and unknown samples. Can not calculate OSR Metrics.")
#
#         if not method:
#             model.log(f"AUROC/OSR/{stage}", np.nan)
#             model.log(f"AUPR-IN/OSR/{stage}", np.nan)
#             model.log(f"AUPR-OUT/OSR/{stage}", np.nan)
#         else:
#             model.log(f"AUROC/OSR/{stage}/{method}", np.nan)
#             model.log(f"AUPR-IN/OSR/{stage}/{method}", np.nan)
#             model.log(f"AUPR-OUT/OSR/{stage}/{method}", np.nan)
#
#     else:
#
#         known = is_known(y)
#         known_unknown = is_known_unknown(y)
#         unknown_unknown = is_unknown_unknown(y)
#
#         known_or_unknown_unknown = is_known(y) | is_unknown_unknown(y)
#
#         try:
#             if unknown_unknown.any():
#                 # see how good we are at distinguishing between known and unkown, but ignore
#                 # known unknowns
#                 scores = score[known_or_unknown_unknown]
#                 labels = known[known_or_unknown_unknown].long()
#                 auroc = metrics.auroc(scores, labels)
#
#                 # AUPR IN
#                 # treat normal class as positive
#                 precision, recall, thresholds = metrics.precision_recall_curve(scores, labels, pos_label=1)
#                 aupr_in = metrics.auc(recall, precision)
#
#                 # AUPR OUT
#                 # treat abnormal class as positive, as described by hendrycks
#                 precision, recall, thresholds = metrics.precision_recall_curve(-scores, 1 - labels, pos_label=1)
#                 aupr_out = metrics.auc(recall, precision)
#
#                 if not method:
#                     model.log(f"AUROC/OSR/{stage}", auroc, prog_bar=prog_bar)
#                     model.log(f"AUPR-IN/OSR/{stage}", aupr_in, prog_bar=prog_bar)
#                     model.log(f"AUPR-OUT/OSR/{stage}", aupr_out, prog_bar=prog_bar)
#                 else:
#                     model.log(f"AUROC/OSR/{stage}/{method}", auroc, prog_bar=prog_bar)
#                     model.log(f"AUPR-IN/OSR/{stage}/{method}", aupr_in, prog_bar=prog_bar)
#                     model.log(f"AUPR-OUT/OSR/{stage}/{method}", aupr_out, prog_bar=prog_bar)
#
#             if known_unknown.any():
#                 log.info(f"Found known unknown: {y[known_unknown].unique()}")
#                 # see how good we are at distinguishing between known known and known unknowns
#                 # this will only be done for methods that train on known unknown data, or if we include
#                 # samples of known unknowns in the validation set
#                 scores = score[known | known_unknown]
#                 labels = known[known | known_unknown].long()
#                 auroc = metrics.auroc(scores, labels)
#
#                 # AUPR IN
#                 precision, recall, thresholds = metrics.precision_recall_curve(scores, labels, pos_label=1)
#                 aupr_in = metrics.auc(recall, precision)
#
#                 # AUPR OUT
#                 precision, recall, thresholds = metrics.precision_recall_curve(-scores, 1-labels, pos_label=1)
#                 aupr_out = metrics.auc(recall, precision)
#
#                 if not method:
#                     model.log(f"AUROC/OSR/{stage}/known", auroc, prog_bar=prog_bar)
#                     model.log(f"AUPR-IN/OSR/{stage}/known", aupr_in, prog_bar=prog_bar)
#                     model.log(f"AUPR-OUT/OSR/{stage}/known", aupr_out, prog_bar=prog_bar)
#                 else:
#                     model.log(f"AUROC/OSR/{stage}/{method}/known", auroc, prog_bar=prog_bar)
#                     model.log(f"AUPR-IN/OSR/{stage}/{method}/known", aupr_in, prog_bar=prog_bar)
#                     model.log(f"AUPR-OUT/OSR/{stage}/{method}/known", aupr_out, prog_bar=prog_bar)
#         except Exception as e:
#             log.error(f"Exception while updating metrics for method {method} in stage {stage}")
#             log.exception(e)
#
#     known = is_known(y)
#     # TODO: we have to handle known unknowns here
#
#     if contains_known(y):
#         if not method:
#             model.log(f"MeanConf/OSR/known/{stage}", score[known].mean(), prog_bar=prog_bar)
#         else:
#             model.log(f"MeanConf/OSR/known/{stage}/{method}", score[known].mean(), prog_bar=prog_bar)
#
#     if contains_unknown(y):
#         if not method:
#             model.log(f"MeanConf/OSR/unknown/{stage}", score[~known].mean(), prog_bar=prog_bar)
#         else:
#             model.log(f"MeanConf/OSR/unknown/{stage}/{method}", score[~known].mean(), prog_bar=prog_bar)


def create_metadata(known, labels, distance=None, centers=None):
    """
    Create metadata for embedding logging
    """
    if distance is not None:
        header = ["label", "known", "distance"]
        data = [[str(l.item()), str(k.item()), str(d.item())] for k, l, d in zip(known, labels, distance)]
    else:
        header = ["label", "known"]
        data = [[str(l.item()), str(k.item())] for k, l in zip(known, labels)]

    return header, data


####################################################
# MISC
####################################################


def _rek_get_hypers(dictionary, obj, current_key):
    """

    """
    # if the given object is a dictionary, process all of its children
    if isinstance(obj, (dict, DictConfig)):
        for key, next_obj in obj.items():
            next_key = f"{current_key}.{key}"
            _rek_get_hypers(dictionary, next_obj, next_key)
    # if it is not (i.e.) it is a single value/list, we add it
    else:
        dictionary[current_key] = obj


def get_hypers(arch_config, optimizer_config, scheduler_config, **kwargs) -> dict:
    """
    Extract hyperparameters to log from the given configs.

    Some people would argue that we could implement this as a single function without
    sife effects if we used a return value instead of passing a "result"-like object, but we suspect that this
    is more performant because we do not have to create a new dict in every recoursion step.
    """
    hypers = dict()
    _rek_get_hypers(hypers, arch_config, "architecture")
    _rek_get_hypers(hypers, optimizer_config, "optimizer")
    _rek_get_hypers(hypers, scheduler_config, "scheduler")

    for key, value in kwargs.items():
        _rek_get_hypers(hypers, value, key)

    return hypers


def set_transformer(dataset, pipeline, target_mapping=None):
    if type(dataset) is Subset:
        log.info("Setting transformer on subset")
        set_transformer(dataset.dataset, pipeline, target_mapping)
        return

    if hasattr(dataset, "transforms"):
        log.info("Setting transformer")
        dataset.transforms = pipeline
    if hasattr(dataset, "transform"):
        log.info("Setting transformer")
        dataset.transform = pipeline

    if target_mapping is not None:
        if not hasattr(dataset, "target_transform"):
            raise ValueError("Dataset does not have target_transform attribute")

        dataset.target_transform = target_mapping


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


def save_embeddings(pl_model, dists, embedding, images, targets, centers=None, tag="default", limit=5000):
    # limit number of saved entries so tensorboard does not crash because of too many sprites
    log.info(f"Saving embeddings")

    indexes = torch.randperm(len(images))[:limit]
    header, data = create_metadata(
        is_known(targets[indexes]),
        targets[indexes],
        distance=torch.min(dists[indexes], dim=1)[0],
        centers=centers
    )

    get_tb_writer(pl_model).add_embedding(
        embedding[indexes],
        metadata=data,
        global_step=pl_model.global_step,
        metadata_header=header,
        label_img=images[indexes],
        tag=tag)
