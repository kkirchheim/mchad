import logging

import pytorch_lightning as pl
from src.utils import save_embeddings
import torch

log = logging.getLogger(__name__)


class SaveEmbeddings(pl.callbacks.Callback):
    """ """

    def __init__(self, use_in_val=True, use_in_test=True, limit=10000, **kwargs):
        self.use_in_val = use_in_val
        self.use_in_test = use_in_test
        self.limit = limit

    def save_embeddings(self, pl_module, outputs, batch, stage):
        if type(batch) is list and type(batch[0]) is list:
            # we are in multi-training-set mode
            batch = torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])

        x, y = batch
        save_embeddings(pl_module, embedding=outputs["embedding"],
                        dists=outputs["dists"], images=x, targets=y, tag=f"Images-{stage}", limit=self.limit)

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
