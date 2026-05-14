import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, f_channels):
        super().__init__()
        with open(f_channels, "r") as f:
            channels = [int(c.strip()) for c in f.readlines()]

        # Encoder
        self.inc = DoubleConv(n_channels, channels[0], channels[1])

        self.down1 = Down(channels[1], channels[2], channels[3])
        self.down2 = Down(channels[3], channels[4], channels[5])
        self.down3 = Down(channels[5], channels[6], channels[7])
        self.down4 = Down(channels[7], channels[8], channels[9])

        # Skip widths are encoder outputs:
        skip1 = channels[1]  # x1
        skip2 = channels[3]  # x2
        skip3 = channels[5]  # x3
        skip4 = channels[7]  # x4

        # Decoder: each Up is defined by (deep_in_ch, skip_ch, mid_ch, out_ch)
        self.up1 = Up(deep_in_ch=channels[10], skip_ch=skip4, mid_ch=channels[11], out_ch=channels[12], bilinear=False)
        self.up2 = Up(deep_in_ch=channels[13], skip_ch=skip3, mid_ch=channels[14], out_ch=channels[15], bilinear=False)
        self.up3 = Up(deep_in_ch=channels[16], skip_ch=skip2, mid_ch=channels[17], out_ch=channels[18], bilinear=False)
        self.up4 = Up(deep_in_ch=channels[19], skip_ch=skip1, mid_ch=channels[20], out_ch=channels[21], bilinear=False)

        self.outc = OutConv(channels[21], n_classes)

    def forward(self, x):
        x1 = self.inc(x)       # skip1
        x2 = self.down1(x1)    # skip2
        x3 = self.down2(x2)    # skip3
        x4 = self.down3(x3)    # skip4
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)
