import logging
from typing import Any, List

import hydra
import torch
import torchmetrics
from pytorch_lightning import LightningModule

from pytorch_ood.utils import is_known
from pytorch_ood.loss import CrossEntropyLoss

from src.utils import load_pretrained_checkpoint, collect_outputs, outputs_detach_cpu
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class SoftMax(LightningModule):
    """
    Model based on the baseline softmax classifier.

    Implements softmax thresholding as baseline for OOD.
    """

    def __init__(
        self,
        backbone: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        n_classes=10,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function
        self.criterion = CrossEntropyLoss()

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model, kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, logits

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            # we will get one batch from each loader
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss, preds, logits = self.step(batch)

        x, y = batch

        self.log(name="Loss/loss_nll/train", value=loss, on_step=True)

        return outputs_detach_cpu({
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
            "embedding": logits,
        })

    def training_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        log_classification_metrics(self, "train", targets, preds, logits)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, logits = self.step(batch)

        self.log(name="Loss/loss_nll/val", value=loss)
        x, y = batch

        return outputs_detach_cpu({
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
            "embedding": logits,
        })

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        log_classification_metrics(self, "val", targets, predictions, logits)

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, logits = self.step(batch)
        x, y = batch

        return outputs_detach_cpu({
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
            "embedding": logits,
        })

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        log_classification_metrics(self, "test", targets, predictions, logits)
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
