from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CosineSchedulerWrapper(CosineAnnealingWarmRestarts):
    """
    Wrapper for scheduler to ease use with hydra
    """

    def __init__(self, epochs, batches_per_epoch, **kwargs):
        super(CosineSchedulerWrapper, self).__init__(
            T_0=epochs * batches_per_epoch, **kwargs
        )
