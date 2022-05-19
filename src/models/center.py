import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from torch import nn

from pytorch_ood.loss import CenterLoss, CrossEntropy
from pytorch_ood.utils import is_known

from src.utils import load_pretrained_checkpoint
from src.utils.logger import collect_outputs, save_embeddings
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class Center(LightningModule):
    """
    Model based on the Center Loss

    Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(
        self,
        optimizer: dict = None,
        scheduler: dict = None,
        backbone: dict = None,
        weight_center=1.0,
        pretrained=None,
        n_classes=10,
        n_embedding=10,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)
        self.classifier = nn.Linear(n_embedding, n_classes)

        # loss function
        self.criterion = CrossEntropy()

        self.center_loss = CenterLoss(n_classes=n_classes, n_dim=n_embedding)
        self.weight_center = weight_center

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model,  kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        z = self.forward(x)
        d = self.center_loss.calculate_distances(z)
        loss_center = self.center_loss(d, y)
        logits = self.classifier(z)
        loss_nll = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss_center, loss_nll, preds, logits, d, y, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        loss_center, loss_nll, preds, logits, d, targets, embedding = self.step(batch)

        x, y = batch

        loss = self.weight_center * loss_center + loss_nll

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)

        # NOTE: we treat the negative distance as logits
        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "logits": logits,
            "dists": d,
            "embedding": embedding.cpu(),
            "points": x.cpu(),
        }

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds, logits)
       #  save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="val")

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, preds, logits, d, targets, embedding = self.step(batch)

        loss = self.weight_center * loss_center + loss_nll

        self.log(name="Loss/loss_center/val", value=loss_center)
        self.log(name="Loss/loss_nll/val", value=loss_nll)
        self.log(name="Loss/loss/val", value=loss)

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "logits": logits,
            "dists": d,
            "embedding": embedding.cpu(),
            "points": x.cpu(),
        }

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds, logits)
       #  save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="val")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, preds, logits, d, targets, embedding = self.step(batch)
        loss = self.weight_center * loss_center + loss_nll

        x, y = batch

        return {
            "loss": loss,
            "preds": preds,
            "targets": targets,
            "logits": logits,
            "dists": d,
            "embedding": embedding.cpu(),
            "points": x.cpu(),
        }

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "test", targets, preds, logits)
        # save_embeddings(
        #     self,
        #     embedding=embedding,
        #     images=images,
        #     targets=targets,
        #     tag=f"test-{self._test_epoch}",
        # )
        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(self.hparams.scheduler.scheduler, optimizer=opti),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]
