import logging

import torch
import torch.nn as nn
from torch.nn import init

import osr.utils

log = logging.getLogger(__name__)


class I3Loss(nn.Module):
    """
    Running Average Center Loss
    """

    def __init__(self, n_classes, n_embedding, alpha=1, **kwargs):
        """

        :param margin:
        :param alpha:
        """
        super(I3Loss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.n_embedding = n_embedding

        running_centers_sum = torch.empty(
            size=(self.n_classes, self.n_embedding), requires_grad=False
        ).double()
        running_center_counters = torch.zeros(
            size=(self.n_classes, 1), requires_grad=False
        ).long()

        self.register_buffer("running_centers_sum", running_centers_sum)
        self.register_buffer("running_center_counters", running_center_counters)
        self.reset_running_stats()

    @property
    def running_centers(self):
        return (
            self.running_centers_sum.div(self.running_center_counters)
            .clone()
            .detach()
            .requires_grad_(True)
        )

    def reset_running_stats(self):
        init.zeros_(self.running_centers_sum)
        init.zeros_(self.running_center_counters)

    def calculate_centers(self, embeddings, target):
        mu = torch.full(
            size=(self.n_classes, self.n_embedding),
            fill_value=float("NaN"),
            device=embeddings.device,
        )

        for clazz in target.unique(sorted=False):
            mu[clazz] = embeddings[target == clazz].mean(
                dim=0
            )  # all instances of this class

        return mu

    def calculate_spreads(self, mu, embeddings, targets):
        class_spreads = torch.zeros(
            (self.n_classes,), device=embeddings.device
        )  # scalar values

        for clazz in targets.unique(sorted=False):
            class_embeddings = embeddings[
                targets == clazz
            ]  # all instances of this class
            class_spreads[clazz] = torch.norm(class_embeddings - mu[clazz], p=2).sum()

        return class_spreads

    def get_center_distances(self, mu):
        n_centers = mu.shape[0]
        a = mu.unsqueeze(1).expand(n_centers, n_centers, mu.size(1)).double()
        b = mu.unsqueeze(0).expand(n_centers, n_centers, mu.size(1)).double()
        dists = torch.norm(a - b, p=2, dim=2)
        return dists

    def calculate_distances(self, embeddings):
        # FIXME: distances will be invalid if squaring is disables
        distances = osr.utils.torch_get_squared_distances(
            self.running_centers, embeddings
        )
        return distances

    def predict(self, embeddings):
        distances = self.calculate_distances(embeddings)
        return nn.functional.softmin(distances, dim=1)

    def forward(self, embeddings, target):
        batch_classes = torch.unique(target, sorted=False)  # already sorted
        n_instances = embeddings.shape[0]

        if self.training:
            # calculate empirical centers
            mu = self.calculate_centers(embeddings, target)
            for clazz in batch_classes:
                self.running_centers_sum[clazz] += embeddings[target == clazz].sum(
                    dim=0
                )
                self.running_center_counters[clazz] += target.eq(clazz).long().sum()

        else:
            # when testing, use the running empirical class centers
            mu = self.running_centers

        # NOTE: pull embeddings towards (all) running centers
        l_spread = (
            self.calculate_spreads(self.running_centers, embeddings, target).sum()
            / n_instances
        )

        # NOTE: get distance between empirical centers
        # dists = - self.get_center_distances(mu[batch_classes])
        # l_separation = torch.log(1 + dists.exp().triu().sum())

        # NOTE TMPMOD
        dists = -self.get_center_distances(mu[batch_classes]).pow(2)
        l_separation = torch.log(dists.exp().triu().sum())

        return l_spread, self.alpha * l_separation
