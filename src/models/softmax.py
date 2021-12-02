import logging
from typing import Any, List

import hydra
import torch
import torchmetrics
from pytorch_lightning import LightningModule

from osr.utils import is_known
from src.utils.metrics import log_classification_metrics
from src.utils.mine import collect_outputs, save_embeddings

log = logging.getLogger(__name__)


class SoftMax(LightningModule):
    """
    Model based on the normal softmax classifier.

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
        self.criterion = torch.nn.CrossEntropyLoss()

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        # save configurations
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.train_auroc = torchmetrics.AUROC(num_classes=2)
        self.val_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.val_auroc = torchmetrics.AUROC(num_classes=2)
        self.test_acc = torchmetrics.Accuracy(num_classes=n_classes)
        self.test_auroc = torchmetrics.AUROC(num_classes=2)

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

        return loss, preds, logits

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            # we will get one batch from each loader
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss, preds, logits = self.step(batch)

        x, y = batch

        self.log(name="Loss/loss_nll/train", value=loss, on_step=True)

        if is_known(y).any():
            self.train_acc.update(logits[is_known(y)], y[is_known(y)])

        conf = torch.softmax(logits, dim=1).max(dim=1).values
        self.train_auroc.update(conf, is_known(y))

        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
            "embedding": logits.cpu(),
            "points": x.cpu(),
        }

    def training_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        preds = collect_outputs(outputs, "preds")
        logits = collect_outputs(outputs, "logits")
        embedding = collect_outputs(outputs, "embedding")
        images = collect_outputs(outputs, "points")

        log_classification_metrics(self, "train", targets, preds, logits)
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="train")
        try:
            log.info(f"ACC Metric: {self.train_acc.compute()}")
            log.info(f"AUROC Metric: {self.train_auroc.compute()}")
        except ValueError as e:
            pass

        self.train_acc.reset()
        self.test_auroc.reset()

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, logits = self.step(batch)

        self.log(name="Loss/loss_nll/val", value=loss)
        x, y = batch

        if is_known(y).any():
            self.val_acc.update(logits[is_known(y)], y[is_known(y)])

        conf = torch.softmax(logits, dim=1).max(dim=1).values
        self.val_auroc.update(conf, is_known(y))

        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
            "embedding": logits.cpu(),
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
        save_embeddings(self, embedding=embedding, images=images, targets=targets, tag="val")

        try:
            log.info(f"ACC Metric: {self.val_acc.compute()}")
            log.info(f"AUROC Metric: {self.val_auroc.compute()}")
        except ValueError as e:
            log.error(f"{e}")
        self.val_acc.reset()
        self.val_auroc.reset()

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss, preds, logits = self.step(batch)
        x, y = batch

        if is_known(y).any():
            self.test_acc.update(logits[is_known(y)], y[is_known(y)])

        conf = torch.softmax(logits, dim=1).max(dim=1).values
        self.test_auroc.update(conf, is_known(y))

        return {
            "loss": loss,
            "preds": preds,
            "targets": y,
            "logits": logits,
            "embedding": logits.cpu(),
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
        save_embeddings(
            self,
            embedding=embedding,
            images=images,
            targets=targets,
            tag=f"test-{self._test_epoch}",
        )

        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        sched = hydra.utils.instantiate(self.scheduler, optimizer=opti)
        return [opti], [sched]
