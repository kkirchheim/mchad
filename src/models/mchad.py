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
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            backbone: dict = None,
            weight_center=1.0,
            weight_oe=1.0,
            weight_ce=1.0,
            pretrained=None,
            n_classes=10,
            n_embedding=10,
            radius=1.0,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function
        self.center_loss = MCHADLoss(n_classes=n_classes, n_embedding=n_embedding)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_oe = weight_oe
        self.weight_center = weight_center
        self.weight_ce = weight_ce

        self.radius = torch.nn.Parameter(torch.tensor([radius]).float())
        self.radius.requires_grad = False

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

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

            d = (self.radius - distmat[~known]).pow(2).relu()
            loss_out = d.sum(dim=1).mean()
        else:
            loss_out = 0

        preds = torch.argmin(distmat, dim=1)

        return loss_center, loss_nll, loss_out, preds, distmat, y, embedding

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list:
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
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # TODO: make configurable
        opti = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            # momentum=self.hparams.momentum,
            # nesterov=self.hparams.nesterov
        )

        # TODO: make configurable
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=opti,
            T_0=20
        )

        return [opti], [sched]


class MCHADLoss(nn.Module):
    """
    Multi Class Hypersphere Anomaly Detection Loss

    :param n_classes: number of classes.
    :param n_embedding: feature dimension.
    :param magnitude: magnitude of

    :see Implementation: https://github.com/KaiyangZhou/pytorch-center-loss
    :see Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(self, n_classes, n_embedding, magnitude=1, fixed=False):
        super(MCHADLoss, self).__init__()
        self.num_classes = n_classes
        self.feat_dim = n_embedding
        self.magnitude = magnitude

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if fixed:
            self.centers.requires_grad = False

        # In the published code, they initialize centers randomly.
        # This, however, is not a good idea if the loss is used without an additional inter-class-discriminability term
        self._init_centers()

    def _init_centers(self):
        if self.num_classes == self.feat_dim:
            torch.nn.init.eye_(self.centers)

            if not self.centers.requires_grad:
                self.centers.mul_(self.magnitude)

            # Orthogonal could also be a good option. this can also be used if the embedding dimensionality is
            # different then the number of classes
            # torch.nn.init.orthogonal_(self.centers, gain=10)
        else:
            torch.nn.init.normal_(self.centers)

            if self.magnitude != 1:
                log.warning(f"Not applying magnitude parameter.")

    def forward(self, distmat, labels) -> torch.Tensor:
        """
        :param: x: feature matrix with shape (batch_size, feat_dim).
        :param labels: ground truth labels with shape (batch_size).
        """
        # distmat = self.pairwise_distances(x, self.centers)

        classes = torch.arange(self.num_classes).long().to(distmat.device)
        labels = labels.unsqueeze(1).expand(distmat.size(0), self.num_classes)
        mask = labels.eq(classes.expand(distmat.size(0), self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / distmat.size(0)

        return loss

    def calculate_distances(self, embeddings: torch.Tensor):
        return self.pairwise_distances(embeddings, self.centers)

    @staticmethod
    def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.

        See https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)

    def predict(self, embeddings):
        distances = self.calculate_distances(embeddings)
        return nn.functional.softmin(distances, dim=1)
