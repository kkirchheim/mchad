from abc import ABC, abstractmethod

import torch


class Method(ABC):
    """
    Abstract Base Class for a method
    """

    @abstractmethod
    def fit(self, data_loader):
        """
        Fit the model to a dataset. Some methods require this.

        :param data_loader: dataset to fit on. This is usually the training dataset.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x) -> torch.Tensor:
        """
        Calculates outlier scores.

        :param x: batch of data
        :return: outlier scores for points
        """
        raise NotImplementedError
