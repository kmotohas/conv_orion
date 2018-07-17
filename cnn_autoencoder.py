#!/usr/bin/env python

"""
convolution:
    x' = 1 + (x - k + 2p) / s
"""
from torch import nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),  # (b, 3, w, h) -> (b, 32, w, h)
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (b, 32, w, h) -> (b, 32, w, h)
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3),  # (b, 32, w, h) -> (b, 16, 32, 32)
            nn.Relu(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (b, 16, 32, 32) -> (b, 16, 16, 16)
        )
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.Relu(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

