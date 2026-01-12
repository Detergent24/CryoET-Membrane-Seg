"""
Minimal 2D U-Net for 2.5D cryo-ET membrane segmentation
"""

import torch
import torch.nn as nn
import config


class ConvBlock(nn.Module):
    """Two 3x3 convs with ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(config.K_SLICES, base)           # P
        self.pool1 = nn.MaxPool2d(2)                 # P/2

        self.enc2 = ConvBlock(base, base * 2)        # P/2
        self.pool2 = nn.MaxPool2d(2)                 # P/4

        self.enc3 = ConvBlock(base * 2, base * 4)    # P/4
        self.pool3 = nn.MaxPool2d(2)                 # P/8

        self.enc4 = ConvBlock(base * 4, base * 8)    # P/8
        self.pool4 = nn.MaxPool2d(2)                 # P/16

        # Bottleneck
        self.bottleneck = ConvBlock(base * 8, base * 16)  # P/16

        # Decoder (upsample + concat skip + conv)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)   # P/8
        self.dec4 = ConvBlock(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)    # P/4
        self.dec3 = ConvBlock(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)    # P/2
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)        # P
        self.dec1 = ConvBlock(base * 2, base)

        # Output logits
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        # Decoder: upsample, concat skip, conv
        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        res = self.out(d1)

        return res


