from torch import nn
from torch.nn import functional as F

import logging


log = logging.getLogger(__name__)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True, dropout=0.3, use_bn=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        # self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=bias)
        # self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x_ = F.relu(x)
        x = self.bn2(self.conv2(x_))
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.bn3(self.conv3(x))
        # x = F.relu(x) + x_
        return x + x_


class ResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, filters=[32,32,32,64,64,64,64,128,128,128], dropout=0.3):
        """

        """
        super(ResNet, self).__init__()
        self.model = nn.Sequential(
            ResBlock(in_channels, filters[0]),
            *[ResBlock(filters[i], filters[i+1], dropout=dropout) for i in range(len(filters) -1)],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[-1], num_classes)
            )

    def forward(self, x):
        return self.model(x)


class WrappedResNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, width=16, depth=3, drop_rate=0.3):
        """

        """
        super(WrappedResNet, self).__init__()
        filters = [width] * depth
        self.model = nn.Sequential(
            ResBlock(in_channels, filters[0]),
            *[ResBlock(filters[i], filters[i+1], dropout=drop_rate) for i in range(len(filters) -1)],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(filters[-1], num_classes)
            )

    def forward(self, x):
        return self.model(x)
