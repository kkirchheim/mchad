import logging
from typing import Any, List

import torch
from torch import nn
import numpy as np
from pytorch_lightning import LightningModule
import hydra

import src.utils.mine as myutils
from src.utils.mine import save_embeddings, collect_outputs
from osr.utils import is_known
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class MCHAD(LightningModule):
    """
    """

    def __init__(
            self,
            backbone: dict = None,
            optimizer: dict = None,
            scheduler: dict = None,
            weight_center=1.0,
            weight_oe=1.0,
            weight_ce=1.0,
            n_classes=10,
            n_embedding=10,
            radius=1.0,
            pretrained=None,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function
        self.center_loss = MchadCenterLoss(n_classes=n_classes, n_embedding=n_embedding)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_oe = weight_oe
        self.weight_center = weight_center
        self.weight_ce = weight_ce

        # radius of the spheres. These are fixed, so they do not require gradients
        self.radius = torch.nn.Parameter(torch.tensor([radius]).float())
        self.radius.requires_grad = False

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        # save configurations
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        embedding = self.forward(x)

        distmat = self.center_loss.calculate_distances(embedding)

        if known.any():
            loss_center = self.center_loss(distmat[known], y[known])

            # calculates softmin
            loss_nll = self.ce_loss(-distmat[known], y[known])
        else:
            loss_nll = 0
            loss_center = 0

        if (~known).any():
            # will give squared distance
            d = (self.radius.pow(2) - distmat[~known].pow(2)).relu()
            loss_out = d.sum(dim=1).sum()  # sum over classes, then sum over batch
        else:
            loss_out = 0

        preds = torch.argmin(distmat, dim=1)

        return loss_center, loss_nll, loss_out, preds, distmat, y, embedding

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        x, y = batch

        loss = self.weight_center * loss_center + self.weight_ce * loss_nll + self.weight_oe * loss_out

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=loss_out, on_step=True)

        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": targets, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds, -dists)
        save_embeddings(self, dists, embedding, images, targets, tag="train", centers=self.center_loss.centers)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        loss = self.weight_center * loss_center + self.weight_ce * loss_nll + self.weight_oe * loss_out
        x, y = batch

        self.log(name="Loss/loss_center/val", value=loss_center)
        self.log(name="Loss/loss_nll/val", value=loss_nll)
        self.log(name="Loss/loss_out/val", value=loss_out)
        self.log(name="Loss/loss/val", value=loss)

        if batch_idx > 1 and batch_idx % 1000 == 0:
            # x, y = batch
            # myutils.get_tb_writer(self).add_images("image/val", x, global_step=self.global_step)
            myutils.log_weight_hists(self)

        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": targets, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds, -dists)
        save_embeddings(self, dists, embedding, images, targets, tag="val")

        # save centers
        save_embeddings(self,
                        embedding=self.center_loss.centers,
                        targets=np.arange(self.center_loss.centers.shape[0]),
                        tag="centers")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        loss = self.weight_center * loss_center + self.weight_ce * loss_nll + self.weight_oe * loss_out

        x, y = batch

        return {"loss": loss, "preds": preds, "targets": targets, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, preds, -dists)
        save_embeddings(self, dists, embedding, images, targets, tag=f"test-{self._test_epoch}")
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        sched = hydra.utils.instantiate(self.scheduler, optimizer=opti)
        return [opti], [sched]



class MchadCenterLoss(nn.Module):
    """
    Multi Class Hypersphere Anomaly Detection Loss (center loss component)

    :param n_classes: number of classes.
    :param n_embedding: feature dimension.
    """

    def __init__(self, n_classes, n_embedding):
        super(MchadCenterLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_embedding
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        torch.nn.init.normal_(self.centers)

    def forward(self, distmat, labels) -> torch.Tensor:
        """
        Calculate loss function.

        :param distmat: matrix with distances
        :param labels: ground truth labels with shape (batch_size).
        """
        classes = torch.arange(self.num_classes).long().to(distmat.device)
        labels = labels.unsqueeze(1).expand(distmat.size(0), self.num_classes)
        mask = labels.eq(classes.expand(distmat.size(0), self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / distmat.size(0)

        return loss

    def calculate_distances(self, embeddings: torch.Tensor):
        return self._pairwise_distances(embeddings, self.centers)

    @staticmethod
    def _pairwise_distances(x, y=None):
        """
        Calculate pairwise distance by quadratic expansion.

        :param x: is a Nxd matrix
        :param y:  Mxd matrix

        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

        See https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

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
