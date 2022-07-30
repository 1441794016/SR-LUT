import torch
import torch.nn as nn


class SRNet(nn.Module):
    def __init__(self, r):
        super(SRNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=1, padding=0),
            # Due to the limited RF size, each color channel has to be processed independently.
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=r*r, kernel_size=1, stride=1, padding=0),
        )
        self.pixel_shuffle = nn.PixelShuffle(r)

    def forward(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        return out
