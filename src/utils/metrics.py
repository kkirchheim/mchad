"""

"""
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from pytorch_lightning.utilities import rank_zero_only

from pytorch_ood.utils import (
    contains_known,
    contains_known_and_unknown,
    contains_unknown,
    is_known,
    is_unknown,
)

log = logging.getLogger(__name__)


def log_metric(model, value, task, stage, metric, method=None, **kwargs):
    """
    Specifies our logging format
    """
    if method:
        model.log(f"{method}/{task}/{metric}/{stage}", value, **kwargs)
    else:
        model.log(f"{task}/{metric}/{stage}", value, **kwargs)


@rank_zero_only
def log_classification_metrics(model: pl.LightningModule, stage, y, y_hat, logits=None):
    if contains_known(y):
        known_idx = is_known(y)
        acc = metrics.accuracy(y_hat[known_idx], y[known_idx])
        log.info(
            f"Accuracy: {acc:.2%} with {known_idx.sum()} known and {(~known_idx).sum()} unknown"
        )
        model.log(f"Accuracy/{stage}", acc, prog_bar=True)

        if logits is not None:
            p = F.log_softmax(logits, dim=1)
            nll = F.nll_loss(p[known_idx], y[known_idx])
            model.log(f"NLL/{stage}", nll)
    else:
        log.warning(
            "Passed data does not contain known samples. Can not calculate Classification Metrics."
        )

        model.log(f"Accuracy/{stage}", np.nan)

        if logits is not None:
            model.log(f"Loss/{stage}", np.nan)

    return

