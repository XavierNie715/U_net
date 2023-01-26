""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torchsummary import summary


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.down4 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down5 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16 // factor, bilinear)
        self.up5 = Up(16, 8, bilinear)
        self.outc = OutConv(8, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        logits = self.outc(x)
        return logits


# todo: 加一个测试网络参数对齐的功能
if __name__ == '__main__':
    unet = UNet(n_channels=2, n_classes=1, bilinear=False)
    x = torch.randn(1, 2, 789, 113)
    y = unet(x)
    # y = ...
    # summary(unet, input_size=(2, 789, 113))
