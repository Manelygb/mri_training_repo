import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(conv => BN => ReLU) * 2  with attribute name `double_conv` for ckpt keys."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """MaxPool then DoubleConv. Keeps `maxpool_conv` for ckpt keys."""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, mid_ch, out_ch),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Decoder block with explicit channel control:
      - up: ConvTranspose2d(deep_in_ch -> skip_ch)
      - conv: DoubleConv(2*skip_ch -> mid_ch -> out_ch)

    This architecture can represent pruned models where skip_ch != deep_in_ch//2.
    """
    def __init__(self, deep_in_ch: int, skip_ch: int, mid_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()

        if bilinear:
            # If you ever switch to bilinear, you must still map deep channels to skip_ch
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(deep_in_ch, skip_ch, kernel_size=1, bias=False),
            )
        else:
            # ConvTranspose2d weight shape: [in_channels, out_channels, k, k]
            self.up = nn.ConvTranspose2d(deep_in_ch, skip_ch, kernel_size=2, stride=2)

        # After up: x1 has skip_ch channels, x2 has skip_ch channels => concat => 2*skip_ch
        self.conv = DoubleConv(2 * skip_ch, mid_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            (diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2),
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
