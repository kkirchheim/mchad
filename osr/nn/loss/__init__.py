"""
Loss functions used in OSR, implemented as ``torch.nn.Module``.

ConfidenceLoss
----------------------------------------------
..  autoclass:: osr.nn.loss.ConfidenceLoss
    :members:


CACLoss
----------------------------------------------

..  autoclass:: osr.nn.loss.CACLoss
    :members:

..  automodule:: osr.nn.loss.cac
    :members:
        rejection_score

IILoss
----------------------------------------------
..  autoclass:: osr.nn.loss.IILoss
    :members:

CenterLoss
----------------------------------------------
.. autoclass:: osr.nn.loss.CenterLoss
    :members:

TripletLoss
----------------------------------------------
.. autoclass:: osr.nn.loss.TripletLoss
    :members:

"""

from .cac import CACLoss
from .center import CenterLoss
from .conf import ConfidenceLoss
from .ii import IILoss
from .triplet import TripletLoss
