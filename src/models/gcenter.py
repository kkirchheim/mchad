import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from pytorch_ood.loss import CenterLoss, CrossEntropyLoss
from pytorch_ood.utils import is_known
from torch import nn

from src.utils import load_pretrained_checkpoint, outputs_detach_cpu, collect_outputs
from src.utils.metrics import log_classification_metrics
from .mchad import CenterRegularizationLoss

log = logging.getLogger(__name__)


class GCenter(LightningModule):
    """
    Generalized version of the center loss

    Paper: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(
        self,
        optimizer: dict = None,
        scheduler: dict = None,
        backbone: dict = None,
        weight_center=0.5,
        weight_oe=0.0005,
        weight_ce=1.5,
        margin=1,
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

        # loss function components
        self.soft_margin_loss = CenterLoss(n_classes=n_classes, n_dim=n_embedding)
        self.nll_loss = CrossEntropyLoss()
        # since we use a soft margin loss, the "radius" of the spheres is the margin
        self.regu_loss = CenterRegularizationLoss(margin=margin)

        self.weight_center = weight_center
        self.weight_oe = weight_oe
        self.weight_nll = weight_ce

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model, kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        z = self.forward(x)

        d = self.soft_margin_loss.calculate_distances(z)
        logits = self.classifier(z)
        loss_center = self.soft_margin_loss(d, y)
        loss_nll = self.nll_loss(logits, y)
        loss_out = self.regu_loss(d, y)

        y_hat = torch.argmin(d, dim=1)

        return loss_center, loss_nll, loss_out, y_hat, logits, z, d

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        l_center, l_nll, l_out, preds, logits, z, d = self.step(batch)

        x, y = batch

        loss = (
            self.weight_center * l_center
            + self.weight_nll * l_nll
            + self.weight_oe * l_out
        )

        self.log(name="Loss/loss_center/train", value=l_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=l_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=l_out, on_step=True)

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": preds,
                "targets": y,
                "logits": logits,
                "dists": d,
                "embedding": z,
            }
        )

    def training_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        log_classification_metrics(self, "train", targets, preds, logits)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        l_center, l_nll, l_out, preds, logits, z, d = self.step(batch)

        loss = (
            self.weight_center * l_center
            + self.weight_nll * l_nll
            + self.weight_oe * l_out
        )

        self.log(name="Loss/loss_center/val", value=l_center)
        self.log(name="Loss/loss_nll/val", value=l_nll)
        self.log(name="Loss/loss_out/val", value=l_out)
        self.log(name="Loss/loss/val", value=loss)

        x, y = batch
        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": preds,
                "targets": y,
                "logits": logits,
                "dists": d,
                "embedding": z,
            }
        )

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        log_classification_metrics(self, "val", targets, preds, logits)

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        l_center, l_nll, l_out, preds, logits, z, d = self.step(batch)

        loss = (
            self.weight_center * l_center
            + self.weight_nll * l_nll
            + self.weight_oe * l_out
        )

        x, y = batch

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": preds,
                "targets": y,
                "logits": logits,
                "embedding": z,
                "dists": d,
            }
        )

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        log_classification_metrics(self, "test", targets, preds, logits)

        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(
                self.hparams.scheduler.scheduler, optimizer=opti
            ),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]
