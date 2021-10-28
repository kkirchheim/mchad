import logging
from typing import Any, List

import torch
from torch import nn
from pytorch_lightning import LightningModule
import hydra

import src.utils.mine as myutils
from osr.utils import is_known
from osr.nn.loss import CenterLoss
from src.utils.mine import save_embeddings, collect_outputs

from src.utils.metrics import log_classification_metrics
from src.utils.mine import create_metadata


log = logging.getLogger(__name__)


class SoftMax(LightningModule):
    """
    Model based on the normal softmax classifier
    """

    def __init__(
            self,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            backbone: dict = None,
            pretrained=None,
            n_classes=10,
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        logits = self.forward(x)

        if known.any():
            loss = self.criterion(logits[known], y[known])
        else:
            loss = 0

        preds = torch.argmax(logits, dim=1)

        return loss, preds, logits, y, logits

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        loss, preds, logits, targets, embedding = self.step(batch)

        x, y = batch

        self.log(name="Loss/loss_nll/train", value=loss, on_step=True)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def training_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds, logits)
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="train")

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, logits, targets, embedding = self.step(batch)

        self.log(name="Loss/loss_nll/val", value=loss)
        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds, logits)
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="val")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, logits, targets, embedding = self.step(batch)

        x, y = batch

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, preds, logits)
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag=f"test-{self._test_epoch}")
        self._test_epoch += 1

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # TODO: make configurable
        # NOTE: 27.10.21 - testing if results are better with SDG with nesterov
        opti = torch.optim.SGD(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
            momentum=0.9, nesterov=True
        )

        # TODO: make configurable
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=opti,
            T_0=20
        )

        return [opti], [sched]
