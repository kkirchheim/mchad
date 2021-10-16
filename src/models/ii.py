import logging
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
import hydra

import src.utils.mine as myutils
from src.utils.mine import save_embeddings, collect_outputs
from src.osr.utils import is_known
from src.utils.metrics import log_classification_metrics
from src.utils.mine import create_metadata
from src.osr.nn.loss import IILoss

log = logging.getLogger(__name__)


class IIModel(LightningModule):
    """
    """

    def __init__(
            self,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            backbone: dict = None,
            n_classes=10,
            n_embedding=10,
            magnitude=1,
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
        if batch_idx > 1 and batch_idx % 1000 == 0:
            x, y = batch
            # myutils.get_tb_writer(self).add_images("image/train", x, global_step=self.global_step)
            myutils.log_weight_hists(self)

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
        self._save_embeddings(dists, embedding, images, targets, tag="train")

    def _save_embeddings(self, dists, embedding, images, targets, tag="default", limit=5000):
        # limit number of saved entries so tensorboard does not crash because of too many sprites
        log.info(f"Saving embeddings")

        indexes = torch.randperm(len(images))[:limit]
        header, data = create_metadata(
            is_known(targets[indexes]),
            targets[indexes],
            distance=torch.min(dists[indexes], dim=1)[0],
            centers=self.ii_loss.centers
        )

        myutils.get_tb_writer(self).add_embedding(
            embedding[indexes],
            metadata=data,
            global_step=self.global_step,
            metadata_header=header,
            label_img=images[indexes],
            tag=tag)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        intra_spread, inter_separation, preds, dists, y, embedding = self.step(batch)

        loss = intra_spread + inter_separation

        self.log(name="Loss/intra_spread/val", value=intra_spread)
        self.log(name="Loss/inter_separation/val", value=inter_separation)

        if batch_idx > 1 and batch_idx % 1000 == 0:
            # x, y = batch
            # myutils.get_tb_writer(self).add_images("image/val", x, global_step=self.global_step)
            myutils.log_weight_hists(self)

        x, y = batch
        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": y, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def _collect_outputs(self, outputs: List[Any], key) -> torch.Tensor:
        if type(outputs) is list:
            # multiple data loaders
            # i have no idea when which case hits ...
            if type(outputs[0]) is list:
                l = []
                for output in outputs:
                    l.extend([o for o in output])
                return torch.cat(l)
            elif type(outputs[0]) is dict:
                return torch.cat([output[key] for output in outputs])
            else:
                l = []
                for output in outputs:
                    l.extend([o for o in output])
                return torch.cat(l)
        else:
            return torch.cat([output[key] for output in outputs])

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
        self._save_embeddings(dists, embedding, images, targets, tag=f"test-{self._test_epoch}")
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
            T_0=100
        )

        return [opti], [sched]
