import logging
import torch
from torch import nn
import torch.nn.functional as F
from src.utils import mine as utils

log = logging.getLogger(__name__)


class OutlierExposureLoss(nn.Module):
    """
    DEEP ANOMALY DETECTION  WITH OUTLIER EXPOSURE

    https://arxiv.org/pdf/1812.04606v1.pdf
    """

    def __init__(self, num_classes, lmbda=1):

        super(OutlierExposureLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_ = lmbda

    def forward(self, logits, target) -> torch.Tensor:
        assert (logits.shape[1] == self.num_classes)

        known = utils.is_known(target)

        logits = F.log_softmax(logits)

        if utils.contains_known(target):
            loss_ce = F.cross_entropy(logits[known], target[known])
        else:
            log.warning(f"No In-Distribution Samples")
            loss_ce = 0

        if utils.contains_unknown(target):
            unity = torch.ones(size=(logits[~known].shape[0], self.num_classes)) / self.num_classes
            unity = unity.to(logits.device)
            loss_oe = F.kl_div(logits[~known], unity, log_target=False, reduction="sum")
        else:
            log.warning(f"No Outlier Samples")
            loss_oe = 0

        return loss_ce, self.lambda_ * loss_oe


class EntropicOpenSetLoss(nn.Module):
    def __init__(self, num_classes):
        pass


