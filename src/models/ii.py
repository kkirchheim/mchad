import logging
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
import hydra

from osr.utils import is_known
from osr.nn.loss import IILoss

from src.utils.mine import save_embeddings, collect_outputs
from src.utils.metrics import log_classification_metrics

log = logging.getLogger(__name__)


class IIModel(LightningModule):
    """
    Model based on *Learning a neural network based representation for open set recognition*.

    :see Paper: https://arxiv.org/pdf/1802.04365.pdf
    """

    def __init__(
            self,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            backbone: dict = None,
            n_classes=10,
            n_embedding=10,
            weight_sep=1.0, # default, as in the original paper
            **kwargs
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        self.ii_loss = IILoss(n_classes=n_classes, n_embedding=n_embedding)

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        known = is_known(y)
        embedding = self.forward(x)

        dists = self.ii_loss.calculate_distances(embedding)

        if known.any():
            intra_spread, inter_separation = self.ii_loss(embedding[known], y[known])
        else:
            intra_spread, inter_separation = 0, 0

        preds = torch.argmin(dists, dim=1)

        return intra_spread, inter_separation, preds, dists, y, embedding

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        intra_spread, inter_separation, preds, dists, y, embedding = self.step(batch)

        x, y = batch

        # NOTE: weighting is not in the original paper
        loss = intra_spread + 100 * inter_separation

        self.log(name="Loss/intra_spread/train", value=intra_spread, on_step=True)
        self.log(name="Loss/inter_separation/train", value=inter_separation, on_step=True)

        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "logits": -dists, "dists": dists,
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
        intra_spread, inter_separation, preds, dists, y, embedding = self.step(batch)

        loss = intra_spread + inter_separation

        self.log(name="Loss/intra_spread/val", value=intra_spread)
        self.log(name="Loss/inter_separation/val", value=inter_separation)

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "logits": -dists, "dists": dists,
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
        intra_spread, inter_separation, preds, dists, y, embedding = self.step(batch)

        loss = intra_spread + inter_separation

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "logits": -dists, "dists": dists,
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
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

        # TODO: make configurable
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=opti,
            T_0=20
        )

        return [opti], [sched]
