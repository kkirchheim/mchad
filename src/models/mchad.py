import logging
from typing import Any, List

import hydra
import torch
from pytorch_lightning import LightningModule
from pytorch_ood.loss import CenterLoss, CrossEntropyLoss
from pytorch_ood.utils import is_known, is_unknown
from torch import nn

from src.utils import (
    collect_outputs,
    get_tensorboard,
    load_pretrained_checkpoint,
    log_classification_metrics,
    outputs_detach_cpu,
)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


log = logging.getLogger(__name__)


class MCHAD(LightningModule):
    """
    Multi Class Hypersphere Anomaly Detection model.

    Uses a radius of 0 for the hyperspheres by default.
    """

    def __init__(
        self,
        backbone: dict = None,
        optimizer: dict = None,
        scheduler: dict = None,
        weight_center=0.5,
        weight_oe=0.0005,
        weight_ce=1.5,
        n_classes=10,
        n_embedding=10,
        margin=1.0,
        radius=0.0,
        save_embeds=False,
        pretrained=None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = hydra.utils.instantiate(backbone)

        # loss function components: center loss, cross-entropy and regularization
        self.soft_margin_loss = CenterLoss(n_classes=n_classes, n_dim=n_embedding, radius=radius)
        self.nll_loss = CrossEntropyLoss()
        self.regu_loss = CenterRegularizationLoss(margin=margin)

        self.weight_oe = weight_oe
        self.weight_center = weight_center
        self.weight_ce = weight_ce

        # count the number of calls to test_epoch_end
        self._test_epoch = 0

        self.save_embeds = save_embeds

        if "pretrained_checkpoint" in kwargs:
            load_pretrained_checkpoint(self.model, kwargs["pretrained_checkpoint"])

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        z = self.forward(x)

        distmat = self.soft_margin_loss.calculate_distances(z)
        loss_center = self.soft_margin_loss(distmat, y)
        # cross-entropy with integrated softmax becomes softmin with e^-x
        loss_nll = self.nll_loss(-distmat, y)
        loss_out = self.regu_loss(distmat, y)

        y_hat = torch.argmin(distmat, dim=1)

        return loss_center, loss_nll, loss_out, y_hat, distmat, z

    def training_step(self, batch: Any, batch_idx: int, **kwargs):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        loss_center, loss_nll, loss_out, y_hat, dists, z = self.step(batch)
        x, y = batch

        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )

        self.log(name="Loss/loss_center/train", value=loss_center, on_step=True)
        self.log(name="Loss/loss_nll/train", value=loss_nll, on_step=True)
        self.log(name="Loss/loss_out/train", value=loss_out, on_step=True)

        if is_known(y).any():
            known_dists = dists[is_known(y)].min(dim=1)[0].mean().item()
            self.log("Distance/known/train", value=known_dists)
        if is_unknown(y).any():
            unknown_dists = dists[is_unknown(y)].min(dim=1)[0].mean().item()
            self.log("Distance/unknown/train", value=unknown_dists)

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": y_hat,
                "targets": y,
                "dists": dists,
                "embedding": z,
            }
        )

    def training_epoch_end(self, outputs: List[Any]):
        target = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "train", target, predictions, -dists)

    def validation_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, y_hat, dists, z = self.step(batch)
        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )
        x, y = batch

        self.log(name="Loss/loss_center/val", value=loss_center)
        self.log(name="Loss/loss_nll/val", value=loss_nll)
        self.log(name="Loss/loss_out/val", value=loss_out)
        self.log(name="Loss/loss/val", value=loss)

        if is_known(y).any():
            known_dists = dists[is_known(y)].min(dim=1)[0].mean().item()
            self.log("Distance/known/val", value=known_dists)
        if is_unknown(y).any():
            unknown_dists = dists[is_unknown(y)].min(dim=1)[0].mean().item()
            self.log("Distance/unknown/val", value=unknown_dists)

        if is_unknown(y).any():
            from src.utils import get_tensorboard

            std = torch.tensor(
                [0.24705882352941176470, 0.24352941176470588235, 0.26156862745098039215]
            )
            mean = torch.tensor(
                [0.49137254901960784313, 0.48235294117647058823, 0.44666666666666666666]
            )
            n = UnNormalize(std=std, mean=mean)
            x_ = torch.stack([n(t) for t in x])
            # model.weight_oe=0.000025 model.weight_center=0.5
            get_tensorboard(self).add_images(
                tag="abc", img_tensor=x_, global_step=self.global_step
            )

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": y_hat,
                "targets": y,
                "dists": dists,
                "embedding": z,
            }
        )

    def validation_epoch_end(self, outputs: List[Any]):
        y = collect_outputs(outputs, "targets")
        predictions = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")

        # log val metrics
        log_classification_metrics(self, "val", y, predictions, -dists)

        if is_known(y).any() and is_known(y).sum() > 1500:
            known_dists = dists[is_known(y)].min(dim=1)[0]
            get_tensorboard(self).add_histogram(
                "Distances/known/val", known_dists, global_step=self.global_step
            )
        if is_unknown(y).any() and is_unknown(y).sum() > 1500:
            unknown_dists = dists[is_unknown(y)].min(dim=1)[0]
            get_tensorboard(self).add_histogram(
                "Distances/unknown/val", unknown_dists, global_step=self.global_step
            )

    def test_step(self, batch: Any, batch_idx: int, *args, **kwargs):
        loss_center, loss_nll, loss_out, y_hat, dists, z = self.step(batch)
        loss = (
            self.weight_center * loss_center
            + self.weight_ce * loss_nll
            + self.weight_oe * loss_out
        )

        x, y = batch

        return outputs_detach_cpu(
            {
                "loss": loss,
                "preds": y_hat,
                "targets": y,
                "dists": dists,
                "embedding": z,
            }
        )

    def test_epoch_end(self, outputs: List[Any]):
        targets = collect_outputs(outputs, "targets")
        prediction = collect_outputs(outputs, "preds")
        dists = collect_outputs(outputs, "dists")
        log_classification_metrics(self, "test", targets, prediction, -dists)

        self._test_epoch += 1

    def configure_optimizers(self):
        opti = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        sched = {
            "scheduler": hydra.utils.instantiate(self.hparams.scheduler.scheduler, optimizer=opti),
            "interval": self.hparams.scheduler.interval,
            "frequency": self.hparams.scheduler.frequency,
        }

        return [opti], [sched]


class CenterRegularizationLoss(nn.Module):
    """
    Regularization Term, uses sum reduction
    """

    def __init__(self, margin):
        """

        :param margin: Margin around centers of the spheres (i.e. including the original radius)
        """
        super(CenterRegularizationLoss, self).__init__()
        self.margin = torch.nn.Parameter(torch.tensor([margin]).float())
        # These are fixed, so they do not require gradients
        self.margin.requires_grad = False

    def forward(self, distmat, target) -> torch.Tensor:
        """
        :param distmat: distance matrix of samples
        :param target: target label of samples
        """
        unknown = is_unknown(target)

        if unknown.any():
            d = (self.margin.pow(2) - distmat[unknown].pow(2)).relu().sum(dim=1)
            # apply reduction
            return d.sum()

        return torch.tensor(0.0, device=distmat.device)
