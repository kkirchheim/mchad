import logging
from typing import Any, List

import torch
from pytorch_lightning import LightningModule
import hydra

import src.utils.mine as myutils
from src.osr.utils import is_known
from src.utils.metrics import log_classification_metrics
from src.utils.mine import create_metadata
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
        self.criterion = torch.nn.CrossEntropyLoss()

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
            loss_nll = self.criterion(-dists[known], y[known])
        else:
            loss_nll = 0
            loss_center = 0

        if ~known.any():
            # will give squared distance
            loss_out = 1 / self.center_loss.calculate_distances(embedding[~known]).sum()
        else:
            loss_out = 0

        preds = torch.argmin(dists, dim=1)

        return loss_center, loss_nll, loss_out, preds, dists, y, embedding

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list:
            results = []

            for b in batch:
                results.append(self.step(b))

            # loss, preds, dists, y, embedding
            loss_center = sum([r[0] for r in results])
            loss_nll = sum([r[1] for r in results])
            loss_out = sum([r[2] for r in results])
            preds = torch.cat([r[3] for r in results])
            dists = torch.cat([r[4] for r in results])
            targets = torch.cat([r[5] for r in results])
            embedding = torch.cat([r[6] for r in results])
            x = torch.cat([b[0] for b in batch])

        else:
            loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)

            x, y = batch

        if batch_idx > 1 and batch_idx % 1000 == 0:
            x, y = batch
            # myutils.get_tb_writer(self).add_images("image/train", x, global_step=self.global_step)
            myutils.log_weight_hists(self)

        # TODO: add weighting
        loss = self.weight_center * loss_center + loss_nll + self.weight_oe * loss_out

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=loss_out, on_step=True)
        # self.log(name="Loss/loss/train", value=loss, on_step=True, prog_bar=True)

        # NOTE: we treat the negative distance as logits
        return {"loss": loss, "preds": preds, "targets": targets, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        targets = self._collect_outputs(outputs, "targets")
        preds = self._collect_outputs(outputs, "preds")
        logits = self._collect_outputs(outputs, "logits")
        dists = self._collect_outputs(outputs, "dists")
        embedding = self._collect_outputs(outputs, "embedding")
        images = self._collect_outputs(outputs, "points")

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
            centers=self.center_loss.centers
        )

        myutils.get_tb_writer(self).add_embedding(
            embedding[indexes],
            metadata=data,
            global_step=self.global_step,
            metadata_header=header,
            label_img=images[indexes],
            tag=tag)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        # if type(batch) is list:
        #     results = []
        #
        #     for b in batch:
        #         results.append(self.step(b))
        #
        #     # loss, preds, dists, y, embedding
        #     loss_center = sum([r[0] for r in results])
        #     loss_nll = sum([r[1] for r in results])
        #     loss_out = sum([r[2] for r in results])
        #     preds = torch.cat([r[3] for r in results])
        #     dists = torch.cat([r[4] for r in results])
        #     targets = torch.cat([r[5] for r in results])
        #     embedding = torch.cat([r[6] for r in results])
        #     x = torch.cat([b[0] for b in batch])
        # else:
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

    def _collect_outputs(self, outputs: List[Any], key) -> torch.Tensor:
        if type(outputs) is list:
            # multiple data loaders
            # i have no idea when which case hits ...
            if type(outputs[0]) is list:
                if type(outputs[0][0]) is dict:
                    l = []
                    for output in outputs:
                        l.extend([o[key] for o in output])
                else:
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
        targets = self._collect_outputs(outputs, "targets")
        preds = self._collect_outputs(outputs, "preds")
        logits = self._collect_outputs(outputs, "logits")
        dists = self._collect_outputs(outputs, "dists")
        embedding = self._collect_outputs(outputs, "embedding")
        images = self._collect_outputs(outputs, "points")

        # log val metrics
        log_classification_metrics(self, "val", targets, preds, logits)
        self._save_embeddings(dists, embedding, images, targets, tag="val")

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, preds, dists, targets, embedding = self.step(batch)
        loss = self.weight_center * loss_center + loss_nll + self.weight_oe * loss_out

        x, y = batch

        return {"loss": loss, "preds": preds, "targets": targets, "logits": -dists, "dists": dists,
                "embedding": embedding.cpu(), "points": x.cpu()}

    def test_epoch_end(self, outputs: List[Any]):
        targets = self._collect_outputs(outputs, "targets")
        preds = self._collect_outputs(outputs, "preds")
        logits = self._collect_outputs(outputs, "logits")
        dists = self._collect_outputs(outputs, "dists")
        embedding = self._collect_outputs(outputs, "embedding")
        images = self._collect_outputs(outputs, "points")

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
            T_0=20
        )

        return [opti], [sched]
