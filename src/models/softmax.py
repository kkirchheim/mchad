import logging
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
import hydra

import src.utils.mine as myutils
from src.utils.mine import save_embeddings, collect_outputs
from src.osr.utils import is_known
from src.utils.metrics import log_classification_metrics

from src.osr.nn.loss import CenterLoss


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
            pretrained=None,
            n_classes=10,
            n_embedding=10,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.center_loss = CenterLoss(n_classes=n_classes, n_embedding=n_embedding)
        self.weight_oe = weight_oe
        self.weight_center = weight_center

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        embedding = self.forward(x)

        dists = self.center_loss.calculate_distances(embedding)

        if known.any():
            loss_center = self.center_loss(embedding[known], y[known])

            # calculates softmin
            loss_nll = self.ce_loss(-dists[known], y[known])
        else:
            loss_nll = 0
            loss_center = 0

        if ~known.any():
            # will give squared distance
            loss_out = (1 / self.center_loss.calculate_distances(embedding[~known])).sum(dim=1).sum()
        else:
            loss_out = 0

        preds = torch.argmin(dists, dim=1)

        return loss_center, loss_nll, loss_out, preds, dists, y, embedding

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        # if type(batch) is list:
        #     batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        x, y = batch

        # if batch_idx > 1 and batch_idx % 1000 == 0:
        #     x, y = batch
        #     # myutils.get_tb_writer(self).add_images("image/train", x, global_step=self.global_step)
        #     myutils.log_weight_hists(self)

        loss = self.weight_center * loss_center + loss_nll + self.weight_oe * loss_out

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=loss_out, on_step=True)

        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": targets, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds, logits)
        save_embeddings(self, dists, embedding, images, targets, tag="train")


    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):

        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        loss = self.weight_center * loss_center + loss_nll + self.weight_oe * loss_out
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
        logits = collect_outputs(outputs, "logits")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds, logits)
        save_embeddings(self, dists, embedding, images, targets, tag="val")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        loss = self.weight_center * loss_center + loss_nll + self.weight_oe * loss_out

        x, y = batch

        return {"loss": loss, "preds": preds, "targets": targets, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        dists = collect_outputs(outputs, "dists")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, preds, logits)
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
            T_0=10
        )

        return [opti], [sched]
