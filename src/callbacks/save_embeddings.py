import logging

import pytorch_lightning as pl
from src.utils import save_embeddings
import torch
from pytorch_ood.utils import TensorBuffer

log = logging.getLogger(__name__)


class SaveEmbeddings(pl.callbacks.Callback):
    """ """

    def __init__(self, use_in_val=True, use_in_test=True, limit=10000, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.limit = limit

        self.test_buffer = TensorBuffer()
        self.val_buffer = TensorBuffer()
        self.test_counter = 0

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.use_in_val:
            dists = self.val_buffer["dists"]
            embeddings = self.val_buffer["embedding"]
            x = self.val_buffer["x"]
            y = self.val_buffer["y"]
            save_embeddings(pl_module, embedding=embeddings, dists=dists, images=x, targets=y, tag="images-val")
            self.val_buffer.clear()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.use_in_test:
            dists = self.test_buffer["dists"]
            embeddings = self.test_buffer["embedding"]
            x = self.test_buffer["x"]
            y = self.test_buffer["y"]
            save_embeddings(pl_module, embedding=embeddings,
                            dists=dists, images=x, targets=y, tag=f"images-test-{self.test_counter}")
            self.test_buffer.clear()
            self.test_counter += 1

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if self.use_in_val:
            x, y = batch
            self.val_buffer.append("dists", outputs["dists"])
            self.val_buffer.append("embedding", outputs["embedding"])
            self.val_buffer.append("x", x)
            self.val_buffer.append("y", y)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            x, y = batch
            self.test_buffer.append("dists", outputs["dists"])
            self.test_buffer.append("embedding", outputs["embedding"])
            self.test_buffer.append("x", x)
            self.test_buffer.append("y", y)

