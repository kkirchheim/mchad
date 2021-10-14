import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceLoss(nn.Module):
    """
    Loss proposed in *Learning Confidence for Out-of-Distribution Detection in Neural Networks*.
    The models learns to predict a confidence :math:`c` in addition to the class membership.

    The loss minimized the Negative Log Likelihood for class membership prediction.

    .. math::
        \\mathcal{L}_{NLL} + \\lambda \\mathcal{L}_c = - \\sum_{i=1}^{M} log(p'_{i}) y_i - \\lambda log(c)

        \\text{where} \\quad p_i' = c \\cdot p_i + (1-c) y_i

    :param lmbda: :math:`\\lambda` used to balance terms
    :param eps: Clipping value :math:`\\epsilon` used for numerical stability

    :see Paper: https://arxiv.org/abs/1802.04865.


    .. note::
        * We implemented clipping for numerical stability.
        * This implementation uses mean reduction for batches.
        * The authors additionally use methods proposed by *Enhancing the reliability of out-of-distribution image detection in neural networks*


    """

    def __init__(self, lmbda=1.0, eps=1e-24):
        super(ConfidenceLoss, self).__init__()
        self.lmbda = lmbda
        self.eps = eps

    def forward(self, logits, confidence, target) -> torch.Tensor:
        """
        :param logits: class logits for samples
        :param confidence: predicted confidence for samples
        :param target: labels for samples (not one-hot encoded)
        """
        target_prob_dist = F.one_hot(target, num_classes=logits.size(1))
        prediction = F.softmax(logits, dim=1)

        adjusted_prediction = prediction * confidence + (1 - confidence) * target_prob_dist

        # calculate negative log likelihood
        adjusted_prediction = adjusted_prediction.clamp(self.eps, 1.0)
        loss_nll = -torch.sum(torch.log(adjusted_prediction) * target_prob_dist)

        confidence = confidence.clamp(self.eps, 1.0)
        loss_conf = - torch.log(confidence)
        loss_conf *= self.lmbda

        # NOTE: we use mean as reduction for batches
        loss_conf = loss_conf.mean()
        return loss_nll, loss_conf
