import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

log = logging.getLogger(__name__)


class IILoss(nn.Module):
    """
    II Loss function from *Learning a neural network based representation for open set recognition*.

    :param n_classes: number of classes
    :param n_embedding: embedding dimensionality

    :see Paper: https://arxiv.org/pdf/1802.04365.pdf
    :see Implementation: https://github.com/shrtCKT/opennet

    .. note::
        * We added running centers for online class center estimation so you do not have to pass forward the
            entire dataset after training.
        * The device of the given embedding will be used as device for all calculations.

    """

    def __init__(self, n_classes, n_embedding, **kwargs):

        super(IILoss, self).__init__()
        self.n_classes = n_classes
        self.n_embedding = n_embedding

        # create buffer for centers. those buffers will be updated during training, and are fixed during evaluation
        running_centers = torch.empty(
            size=(self.n_classes, self.n_embedding), requires_grad=False
        ).float()
        num_batches_tracked = torch.empty(size=(1,), requires_grad=False).float()

        self.register_buffer("running_centers", running_centers)
        self.register_buffer("num_batches_tracked", num_batches_tracked)
        self.reset_running_stats()

    @property
    def centers(self):
        return self.running_centers

    def reset_running_stats(self):
        log.info("Reset running stats")
        init.zeros_(self.running_centers)
        init.zeros_(self.num_batches_tracked)

    def calculate_centers(self, embeddings, target):
        mu = torch.full(
            size=(self.n_classes, self.n_embedding),
            fill_value=float("NaN"),
            device=embeddings.device,
        )

        for clazz in target.unique(sorted=False):
            mu[clazz] = embeddings[target == clazz].mean(dim=0)  # all instances of this class

        return mu

    def calculate_spreads(self, mu, embeddings, targets):
        class_spreads = torch.zeros((self.n_classes,), device=embeddings.device)  # scalar values

        # calculate sum of (squared) distances of all instances to the class center
        for clazz in targets.unique(sorted=False):
            class_embeddings = embeddings[targets == clazz]  # all instances of this class
            class_spreads[clazz] = torch.norm(class_embeddings - mu[clazz], p=2).pow(2).sum()

        return class_spreads

    def get_center_distances(self, mu):
        n_centers = mu.shape[0]
        a = mu.unsqueeze(1).expand(n_centers, n_centers, mu.size(1)).float()
        b = mu.unsqueeze(0).expand(n_centers, n_centers, mu.size(1)).float()
        dists = torch.norm(a - b, p=2, dim=2).pow(2)

        # set diagonal elements to "high" value (this value will limit the inter seperation, so cluster
        # do not drift apart infinitely)
        dists[torch.eye(n_centers, dtype=torch.bool)] = 1e24
        return dists

    def calculate_distances(self, embeddings):
        return pairwise_distances(embeddings, self.centers)

    def predict(self, embeddings):
        distances = self.calculate_distances(embeddings)
        return nn.functional.softmin(distances, dim=1)

    def forward(self, embeddings, target) -> torch.Tensor:
        """

        :param embeddings: embeddings of samples
        :param target: label of samples
        """
        batch_classes = torch.unique(target, sorted=False)
        n_instances = embeddings.shape[0]

        if self.training:
            # calculate empirical centers
            mu = self.calculate_centers(embeddings, target)

            # update running mean centers
            cma = (
                mu[batch_classes] + self.running_centers[batch_classes] * self.num_batches_tracked
            )
            self.running_centers[batch_classes] = cma / (self.num_batches_tracked + 1)
            self.num_batches_tracked += 1
        else:
            # when testing, use the running empirical class centers
            mu = self.running_centers

        # calculate sum of class spreads and divide by the number of instances
        intra_spread = self.calculate_spreads(mu, embeddings, target).sum() / n_instances

        # calculate distance between all (present) class centers
        dists = self.get_center_distances(mu[batch_classes])

        # the minimum distance between all class centers is the inter separation
        inter_separation = -torch.min(dists)

        # intra_spread should be minimized, inter_separation maximized
        return intra_spread, inter_separation


def pairwise_distances(x, y=None):
    """
    :param x: Nxd matrix
    :param y:  optional Mxd matrix
    :return: a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.

    :see Web: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3

    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
