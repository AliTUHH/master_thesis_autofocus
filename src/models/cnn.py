# -*- coding: utf-8 -*-
import torch.nn as nn

class AutofocusCNN(nn.Module):
    """Convolutional regressor for estimating axial defocus in centimeters.

    Four convolutional blocks progressively extract spatial fringe features
    while reducing the image resolution from 128 x 128 to 8 x 8. The final
    fully connected head maps those features to one continuous distance value.
    """

    def __init__(self, config):
        super(AutofocusCNN, self).__init__()
        channels = config['conv_channels']
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], 3, padding=1), nn.BatchNorm2d(channels[0]), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], 3, padding=1), nn.BatchNorm2d(channels[1]), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], 3, padding=1), nn.BatchNorm2d(channels[2]), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels[2], channels[3], 3, padding=1), nn.BatchNorm2d(channels[3]), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(channels[3] * 8 * 8, config['fc_units']),
            nn.ReLU(),
            nn.Dropout(config['dropout_rate']),
            # A single linear output represents signed defocus in centimeters.
            nn.Linear(config['fc_units'], 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x).squeeze()
