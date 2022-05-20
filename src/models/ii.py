import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from torch.nn import BatchNorm1d

from pytorch_ood.loss import IILoss

from src.utils import load_pretrained_checkpoint, collect_outputs, outputs_detach_cpu
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class IIModel(LightningModule):
    """
    Model based on *Learning a neural network based representation for open set recognition*.

    :see Paper: https://arxiv.org/pdf/1802.04365.pdf
    """

    def __init__(
        self,
        backbone: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        n_classes=10,
        n_embedding=10,
        weight_sep=100.0,  # weight for separation term., not included in the original paper
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)
        self.ii_loss = IILoss(n_classes=n_classes, n_embedding=n_embedding)
        self.weight_sep = weight_sep
        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        self.bn = BatchNorm1d(n_embedding)

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model, kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        # this implementation uses running average estimates for the centers.
        # these should be reset regularly
        self.ii_loss.centers.reset_running_stats()

    def step(self, batch: Any):
        x, y = batch
        z = self.forward(x)

        # do additional batch norm layer to embedding
        z = self.bn(z)

        intra_spread, inter_separation = self.ii_loss(z, y)
        dists = self.ii_loss.calculate_distances(z)
        preds = torch.argmin(dists, dim=1)

        return intra_spread, inter_separation, preds, dists, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        intra_spread, inter_separation, preds, dists, z = self.step(batch)

        x, y = batch

        # NOTE: weighting is not in the original paper, but it collapses immediately without it
        loss = intra_spread + self.weight_sep * inter_separation

        self.log(name="Loss/intra_spread/train", value=intra_spread, on_step=True)
        self.log(
            name="Loss/inter_separation/train", value=inter_separation, on_step=True
        )

        return outputs_detach_cpu({
            "loss": loss,
            "preds": preds,
            "targets": y,
            "dists": dists,
            "embedding": z,
        })

    def training_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "train", targets, predictions, -dists)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        intra_spread, inter_separation, preds, dists, embedding = self.step(batch)

        loss = intra_spread + self.weight_sep * inter_separation

        self.log(name="Loss/intra_spread/val", value=intra_spread)
        self.log(name="Loss/inter_separation/val", value=inter_separation)

        x, y = batch
        return outputs_detach_cpu({
            "loss": loss,
            "preds": preds,
            "targets": y,
            "dists": dists,
            "embedding": embedding,
        })

    def validation_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "val", targets, predictions, -dists)

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        intra_spread, inter_separation, preds, dists, embedding = self.step(batch)

        loss = intra_spread + self.weight_sep * inter_separation

        x, y = batch
        return outputs_detach_cpu({
            "loss": loss,
            "preds": preds,
            "targets": y,
            "dists": dists,
            "embedding": embedding,
        })

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "test", targets, predictions, -dists)
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
