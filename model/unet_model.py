import torch.nn.functional as F
from model.resnet import resnet34 as resnet
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.backbone = resnet()

        self.up1 = Up(512+256, 256, bilinear)
        self.up2 = Up(256+128, 128, bilinear)
        self.up3 = Up(128+64, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.upc = UpwithCrop()

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.backbone(x)

        logits = self.up1(x5, x4)
        logits = self.up2(logits, x3)
        logits = self.up3(logits, x2)
        logits = self.up4(logits, x1)
        logits = self.upc(logits, x)

        logits = self.outc(logits)
        return logits
