# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F

from unet.unet_parts import *


class AddNoise(nn.Module):
    def __init__(self, mean=0.0, stddev=0.1):
        super(AddNoise, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, input):
        if self.training:
            noise = input.clone().normal_(self.mean, self.stddev)
            return input + noise
        else:
            return input


class UNetNoisy(nn.Module):
    def __init__(self, n_channels):
        super(UNetNoisy, self).__init__()
        self.noise = AddNoise()
        self.low_noise = AddNoise(stddev=0.05)
        self.inc = inconv(n_channels, 10)
        self.down1 = down(10, 20)
        self.down2 = down(20, 30)
        self.down3 = down(30, 40)
        self.up1 = up(70, 30)
        self.up2 = up(50, 20)
        self.up3 = up(30, 10)
        self.outc = outconv(10, n_channels)

    def forward(self, x):
        x = self.noise(x)
        x1 = self.inc(x)
        x1_noisy = self.low_noise(x1)
        x2 = self.down1(x1_noisy)
        x2_noisy = self.low_noise(x2)
        x3 = self.down2(x2_noisy)
        x3_noisy = self.low_noise(x3)
        x4 = self.down3(x3_noisy)
        x4_noisy = self.low_noise(x4)
        x = self.up1(x4_noisy, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return x
        #return F.sigmoid(x)
