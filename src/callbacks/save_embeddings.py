import logging

import pytorch_lightning as pl
from src.utils import save_embeddings

log = logging.getLogger(__name__)


class SaveEmbeddings(pl.callbacks.Callback):
    """ """

    def __init__(self, use_in_val=False, use_in_test=True, limit=10000, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.limit = limit

    def save_embeddings(self, pl_module, batch, outputs, stage):
        x, y = batch
        save_embeddings(pl_module, dists=outputs["dists"], images=x, targets=y, tag=f"Images/{stage}", limit=self.limit)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""
        if self.use_in_val:
            self.save_embeddings(pl_module, outputs, batch, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the test batch ends."""
        if self.use_in_test:
            self.save_embeddings(pl_module, outputs, batch, "val")
