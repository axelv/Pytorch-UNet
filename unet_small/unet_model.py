# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 10)
        self.down1 = down(10, 20)
        self.down2 = down(20, 30)
        self.down3 = down(30, 40)
        self.up1 = up(40, 30)
        self.up2 = up(30, 20)
        self.up3 = up(20, 10)
        self.outc = outconv(10, n_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
        #return F.sigmoid(x)
