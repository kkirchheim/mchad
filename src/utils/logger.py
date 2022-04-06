#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utils mainly used for logging putposes, for example to save embeddings, gradients, weights etc.

"""
import logging
import time
from collections import defaultdict
from typing import Any, List

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.base import LoggerCollection
from torch.utils.tensorboard import SummaryWriter

from oodtk.utils import contains_known, contains_unknown, is_known, is_unknown

log = logging.getLogger(__name__)


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
            writer.add_histogram(tag=f"{prefix}/unknown", values=score[unknown], global_step=epoch)


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
    limit=5000,
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
    indexes = torch.randperm(len(embedding))[:limit]

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
