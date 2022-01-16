import logging
from typing import Any, List

import hydra
import numpy as np
import torch
from osr.utils import is_known, is_unknown
from pytorch_lightning import LightningModule
from torch import nn

from src.utils.logger import collect_outputs, get_tensorboard, save_embeddings
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class MCHAD(LightningModule):
    """
    Multi Class Hypersphere Anomaly Detection model
    """

    def __init__(
        self,
        backbone: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        weight_center=0.5,
        weight_oe=0.1,
        weight_ce=1.0,
        n_classes=10,
        n_embedding=10,
        radius=1.0,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function components
        self.soft_margin_loss = SoftMarginLoss(n_classes=n_classes, z_dim=n_embedding)
        self.nll_loss = nn.CrossEntropyLoss()
        # since we use a soft margin loss, the "radius" of the spheres is the margin
        self.regu_loss = MCHADRegularizationLoss(margin=radius)

        self.weight_oe = weight_oe
        self.weight_center = weight_center
        self.weight_ce = weight_ce

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        z = self.forward(x)

        distmat = self.soft_margin_loss.calculate_distances(z)

        if known.any():
            loss_center = self.soft_margin_loss(distmat[known], y[known])
            # cross-entropy with integrated softmax becomes softmin with e^-x
            loss_nll = self.nll_loss(-distmat[known], y[known])
        else:
            loss_nll = 0
            loss_center = 0

        if (~known).any():
            loss_out = self.regu_loss(distmat[~known])
        else:
            loss_out = 0

        y_hat = torch.argmin(distmat, dim=1)

        return loss_center, loss_nll, loss_out, y_hat, distmat, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss_center, loss_nll, loss_out, y_hat, dists, z = self.step(batch)
        x, y = batch

        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=loss_out, on_step=True)

        if is_known(y).any():
            known_dists = dists[is_known(y)].min(dim=1)[0].mean().item()
            self.log("Distance/known/train", value=known_dists)
        if is_unknown(y).any():
            unknown_dists = dists[is_unknown(y)].min(dim=1)[0].mean().item()
            self.log("Distance/unknown/train", value=unknown_dists)

        return {
            "loss": loss,
            "preds": y_hat,
            "targets": y,
            "dists": dists,
            "embedding": z.cpu(),
            "points": x.cpu(),
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        target = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        z = collect_outputs(outputs, "embedding")
        x = collect_outputs(outputs, "points")

        mu = self.soft_margin_loss.centers
        log_classification_metrics(self, "train", target, predictions, -dists)
        save_embeddings(self, dists, z, x, target, tag="train")

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, y_hat, dists, z = self.step(batch)
        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )
        x, y = batch

        self.log(name="Loss/loss_center/val", value=loss_center)
        self.log(name="Loss/loss_nll/val", value=loss_nll)
        self.log(name="Loss/loss_out/val", value=loss_out)
        self.log(name="Loss/loss/val", value=loss)

        if is_known(y).any():
            known_dists = dists[is_known(y)].min(dim=1)[0].mean().item()
            self.log("Distance/known/val", value=known_dists)
        if is_unknown(y).any():
            unknown_dists = dists[is_unknown(y)].min(dim=1)[0].mean().item()
            self.log("Distance/unknown/val", value=unknown_dists)

        return {
            "loss": loss,
            "preds": y_hat,
            "targets": y,
            "dists": dists,
            "embedding": z.cpu(),
            "points": x.cpu(),
        }

    def validation_epoch_end(self, outputs: List[Any]):
        y = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        z = collect_outputs(outputs, "embedding")
        x = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", y, predictions, -dists)
        save_embeddings(self, dists, z, x, y, tag="val")

        if is_known(y).any() and is_known(y).sum() > 1500:
            known_dists = dists[is_known(y)].min(dim=1)[0]
            get_tensorboard(self).add_histogram(
                "Distances/known/val", known_dists, global_step=self.global_step
            )
        if is_unknown(y).any() and is_unknown(y).sum() > 1500:
            unknown_dists = dists[is_unknown(y)].min(dim=1)[0]
            get_tensorboard(self).add_histogram(
                "Distances/unknown/val", unknown_dists, global_step=self.global_step
            )

        # save centers
        save_embeddings(
            self,
            embedding=self.soft_margin_loss.centers,
            targets=np.arange(self.soft_margin_loss.centers.shape[0]),
            tag="centers",
        )

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, y_hat, dists, z = self.step(batch)
        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )

        x, y = batch

        return {
            "loss": loss,
            "preds": y_hat,
            "targets": y,
            "dists": dists,
            "embedding": z.cpu(),
            "points": x.cpu(),
        }

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        prediction = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        z = collect_outputs(outputs, "embedding")
        x = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, prediction, -dists)
        save_embeddings(self, dists, z, x, targets, tag=f"test-{self._test_epoch}")
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(self.hparams.scheduler.scheduler, optimizer=opti),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]


class MCHADRegularizationLoss(nn.Module):
    """
    Regularization Term, uses sum reduction
    """

    def __init__(self, margin):
        """

        :param margin: Margin around centers of the spheres (i.e. including the original radius)
        """
        super(MCHADRegularizationLoss, self).__init__()
        self.margin = torch.nn.Parameter(torch.tensor([margin]).float())
        # These are fixed, so they do not require gradients
        self.margin.requires_grad = False

    def forward(self, distmat) -> torch.Tensor:
        """
        :param distmat: distance matrix of samples
        """
        d = (self.margin.pow(2) - distmat.pow(2)).relu().sum(dim=1)
        return d.sum()


class SoftMarginLoss(nn.Module):
    """
    Soft margin loss component.
    Uses mean reduction by default.
    Also holds the class centers.

    :param n_classes: number of classes.
    :param z_dim: dimensionality of output space
    """

    def __init__(self, n_classes, z_dim):
        super(SoftMarginLoss, self).__init__()
        self.num_classes = n_classes
        self.z_dim = z_dim
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.z_dim))
        torch.nn.init.normal_(self.centers)

    def forward(self, distmat, target) -> torch.Tensor:
        """
        Calculate center loss component function.

        :param distmat: matrix with distances
        :param target: ground truth labels with shape (batch_size).
        """
        classes = torch.arange(self.num_classes).long().to(distmat.device)
        target = target.unsqueeze(1).expand(distmat.size(0), self.num_classes)
        mask = target.eq(classes.expand(distmat.size(0), self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e12).sum() / distmat.size(0)

        return loss

    def calculate_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate distance of given embeddings to each class center

        :param embeddings: embeddings to calculate distance to
        """
        return self._pairwise_distances(embeddings, self.centers)

    @staticmethod
    def _pairwise_distances(x, y=None) -> torch.Tensor:
        """
        Calculate pairwise distance by quadratic expansion.

        :param x: is a Nxd matrix
        :param y:  Mxd matrix

        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

        :see Implementation: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

        return torch.clamp(dist, 0.0, np.inf)

    def predict(self, embeddings):
        distances = self.calculate_distances(embeddings)
        return nn.functional.softmin(distances, dim=1)
