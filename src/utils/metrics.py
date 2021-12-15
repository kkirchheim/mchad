"""

"""
import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from pytorch_lightning.utilities import rank_zero_only

from osr.utils import (
    contains_known,
    contains_known_and_unknown,
    contains_unknown,
    is_known,
    is_known_unknown,
    is_unknown,
    is_unknown_unknown,
)

log = logging.getLogger(__name__)


def _log(model, value, task, stage, metric, method=None, **kwargs):
    """
    Specifies our logging format
    """
    if method:
        model.log(f"{method}/{task}/{metric}/{stage}", value, **kwargs)
    else:
        model.log(f"{task}/{metric}/{stage}", value, **kwargs)


def fpr_at_tpr(pred, target, k=0.95):
    fpr, tpr, thresholds = metrics.roc(pred, target)
    for fp, tp, t in zip(fpr, tpr, thresholds):
        if tp >= k:
            return fp


def accuracy_at_tpr(pred, target, k=0.95):
    fpr, tpr, thresholds = metrics.roc(pred, target)
    for fp, tp, t in zip(fpr, tpr, thresholds):
        if tp >= k:
            break

    labels = torch.where(pred > t, 1, 0)
    return metrics.accuracy(labels, target)


@rank_zero_only
def log_error_detection_metrics(
    model: pl.LightningModule, score, stage, y, y_hat, method=None, prog_bar=False
):
    """
    Log error-dectection metrics, AUROC and AUPR

    Error Detection Refers to the ability to discriminate correctly classified and misclassified images.
    It is unrelated to OOD/OSR, as it does not consider samples from unknown classes.
    """
    if not contains_known(y):
        log.warning("Passed data does not contain known samples. Can not calculate error Metrics.")

        _log(model, np.nan, "Error", stage, "AUROC", method)
        _log(model, np.nan, "Error", stage, "AUPR-IN", method)
        _log(model, np.nan, "Error", stage, "AUPR-OUT", method)
        _log(model, np.nan, "Error", stage, "MeanConf/incorrect", method)
        _log(model, np.nan, "Error", stage, "MeanConf/correct", method)
        return

    correct_class = (y == y_hat).long()
    known = is_known(y)

    try:
        auroc = metrics.auroc(score[known], correct_class[known])

        # AUPR IN
        precision, recall, thresholds = metrics.precision_recall_curve(
            score[known], correct_class[known], pos_label=1
        )
        aupr_in = metrics.auc(recall, precision)

        # AUPR OUT
        precision, recall, thresholds = metrics.precision_recall_curve(
            -score[known], 1 - correct_class[known], pos_label=1
        )
        aupr_out = metrics.auc(recall, precision)

        fpt_at_95tpr = fpr_at_tpr(score[known], correct_class[known])
        acc_at_95_tpr = accuracy_at_tpr(score[known], correct_class[known])

        _log(model, auroc, "Error", stage, "AUROC", method, prog_bar=prog_bar)
        _log(model, aupr_in, "Error", stage, "AUPR-IN", method, prog_bar=prog_bar)
        _log(model, aupr_out, "Error", stage, "AUPR-OUT", method, prog_bar=prog_bar)
        _log(model, fpt_at_95tpr, "Error", stage, "FPR@95TPR", method, prog_bar=prog_bar)
        _log(model, acc_at_95_tpr, "Error", stage, "ACC@95TPR", method, prog_bar=prog_bar)
        _log(
            model,
            score[known & ~(y == y_hat)].mean(),
            "Error",
            stage,
            "MeanConf/incorrect",
            method,
            prog_bar=prog_bar,
        )
        _log(
            model,
            score[known & (y == y_hat)].mean(),
            "Error",
            stage,
            "MeanConf/correct",
            method,
            prog_bar=prog_bar,
        )

    except Exception as e:
        log.exception(e)
        log.error("Exception while calculating error detection metrics")


@rank_zero_only
def log_uncertainty_metrics(model, score, stage, y, y_hat, method=None, prog_bar=False):
    """
    Log uncertainty metrics, AUROC and AUPR

    Uncertainty refers to the ability to discriminate correctly classified images from
    unknown or misclassified images.
    """
    if not contains_known_and_unknown(y):
        log.warning(
            "Passed data does not contain known and unknown samples. Can not calculate uncertainty Metrics."
        )
        _log(model, np.nan, "Uncertainty", stage, "AUROC", method)
        _log(model, np.nan, "Uncertainty", stage, "AUPR-IN", method)
        _log(model, np.nan, "Uncertainty", stage, "AUPR-OUT", method)
    else:
        try:
            known_and_correct = is_known(y) & (y == y_hat)
            if known_and_correct.any():
                auroc = metrics.auroc(score, known_and_correct)

                # AUPR IN
                precision, recall, thresholds = metrics.precision_recall_curve(
                    score, known_and_correct.long(), pos_label=1
                )
                aupr_in = metrics.auc(recall, precision)

                # AUPR OUT
                precision, recall, thresholds = metrics.precision_recall_curve(
                    -score, 1 - known_and_correct.long(), pos_label=1
                )

                aupr_out = metrics.auc(recall, precision)

                fpt_at_95tpr = fpr_at_tpr(score, known_and_correct.long())
                acc_at_95_tpr = accuracy_at_tpr(score, known_and_correct.long())

                _log(
                    model,
                    auroc,
                    "Uncertainty",
                    stage,
                    "AUROC",
                    method,
                    prog_bar=prog_bar,
                )
                _log(
                    model,
                    aupr_in,
                    "Uncertainty",
                    stage,
                    "AUPR-IN",
                    method,
                    prog_bar=prog_bar,
                )
                _log(
                    model,
                    aupr_out,
                    "Uncertainty",
                    stage,
                    "AUPR-OUT",
                    method,
                    prog_bar=prog_bar,
                )
                _log(
                    model,
                    fpt_at_95tpr,
                    "Uncertainty",
                    stage,
                    "FPR@95TPR",
                    method,
                    prog_bar=prog_bar,
                )
                _log(
                    model,
                    acc_at_95_tpr,
                    "Uncertainty",
                    stage,
                    "ACC@95TPR",
                    method,
                    prog_bar=prog_bar,
                )
            else:
                log.warning("No known and correct points")
        except Exception as e:
            log.exception(e)
            log.error("Exception while calculating uncertainty detection metrics")

    if contains_known(y):
        _log(
            model,
            score[is_known(y)].mean(),
            "Uncertainty",
            stage,
            "MeanConf/known/",
            method,
            prog_bar=prog_bar,
        )

    if contains_unknown(y):
        _log(
            model,
            score[~is_known(y)].mean(),
            "Uncertainty",
            stage,
            "MeanConf/unknown/",
            method,
            prog_bar=prog_bar,
        )


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


@rank_zero_only
def log_osr_metrics(model: pl.LightningModule, score, stage, y, method=None, prog_bar=False):
    """
    Log OSR metrics, AUROC and AUPR etc.

    :param model: the model to log for
    :param score: score for samples. should be a confidence, i.e. high values indicate
        "normal" or "in-distribution" points.
    """

    if not contains_known_and_unknown(y):
        log.warning(
            "Passed data does not contain known and unknown samples. Can not calculate OSR Metrics."
        )
        _log(model, np.nan, "OSR", stage, "AUROC", method)
        _log(model, np.nan, "OSR", stage, "AUPR-IN", method)
        _log(model, np.nan, "OSR", stage, "AUPR-OUT", method)
    else:

        known = is_known(y)
        unknown = is_unknown(y)

        try:
            if contains_known_and_unknown(y):
                # see how good we are at distinguishing between known and unkown, but ignore
                # known unknowns
                scores = score
                labels = known.long()

                auroc = metrics.auroc(scores, labels)

                # AUPR IN
                # treat normal class as positive
                precision, recall, thresholds = metrics.precision_recall_curve(
                    scores, labels, pos_label=1
                )
                aupr_in = metrics.auc(recall, precision)

                # AUPR OUT
                # treat abnormal class as positive, as described by hendrycks
                precision, recall, thresholds = metrics.precision_recall_curve(
                    -score, 1 - labels, pos_label=1
                )
                # precision, recall, thresholds = metrics.precision_recall_curve(scores, labels, pos_label=0)
                aupr_out = metrics.auc(recall, precision)

                fpt_at_95tpr = fpr_at_tpr(score, labels)
                acc_at_95_tpr = accuracy_at_tpr(score, labels)

                _log(model, auroc, "OSR", stage, "AUROC", method, prog_bar=prog_bar)
                _log(model, aupr_in, "OSR", stage, "AUPR-IN", method, prog_bar=prog_bar)
                _log(model, aupr_out, "OSR", stage, "AUPR-OUT", method, prog_bar=prog_bar)
                _log(
                    model,
                    fpt_at_95tpr,
                    "OSR",
                    stage,
                    "FPR@95TPR",
                    method,
                    prog_bar=prog_bar,
                )
                _log(
                    model,
                    acc_at_95_tpr,
                    "OSR",
                    stage,
                    "ACC@95TPR",
                    method,
                    prog_bar=prog_bar,
                )

        except Exception as e:
            log.error(f"Exception while updating OSR metrics for method {method} in stage {stage}")
            log.exception(e)

    if contains_known(y):
        _log(
            model,
            score[is_known(y)].mean(),
            "OSR",
            stage,
            "MeanConf/known/",
            method,
            prog_bar=prog_bar,
        )

    if contains_unknown(y):
        _log(
            model,
            score[~is_known(y)].mean(),
            "OSR",
            stage,
            "MeanConf/unknown/",
            method,
            prog_bar=prog_bar,
        )
