"""U-Net++ (nested U-Net) architecture for binary segmentation."""

import torch
import torch.nn as nn

from train_pipeline.models.base import SegmentationModel
from train_pipeline.models.registry import register_model


@register_model("unet_plusplus")
class UNetPlusPlus(SegmentationModel):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        **kwargs,
    ):
        super().__init__(in_channels, num_classes)
        bc = base_channels

        self.encoder1 = self._conv_block(in_channels, bc)
        self.encoder2 = self._conv_block(bc, bc * 2)
        self.encoder3 = self._conv_block(bc * 2, bc * 4)
        self.encoder4 = self._conv_block(bc * 4, bc * 8)
        self.encoder5 = self._conv_block(bc * 8, bc * 16)

        self.pool = nn.MaxPool2d(2, 2)

        self.up1_0 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.conv1_0 = self._conv_block(bc * 2, bc)

        self.up2_0 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.conv2_0 = self._conv_block(bc * 4, bc * 2)
        self.up1_1 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.conv1_1 = self._conv_block(bc * 3, bc)

        self.up3_0 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.conv3_0 = self._conv_block(bc * 8, bc * 4)
        self.up2_1 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.conv2_1 = self._conv_block(bc * 6, bc * 2)
        self.up1_2 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.conv1_2 = self._conv_block(bc * 4, bc)

        self.up4_0 = nn.ConvTranspose2d(bc * 16, bc * 8, 2, stride=2)
        self.conv4_0 = self._conv_block(bc * 16, bc * 8)
        self.up3_1 = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.conv3_1 = self._conv_block(bc * 12, bc * 4)
        self.up2_2 = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.conv2_2 = self._conv_block(bc * 8, bc * 2)
        self.up1_3 = nn.ConvTranspose2d(bc * 2, bc, 2, stride=2)
        self.conv1_3 = self._conv_block(bc * 5, bc)

        self.final = nn.Conv2d(bc, num_classes, 1)

    @staticmethod
    def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        e5 = self.encoder5(self.pool(e4))

        d1_0 = self.conv1_0(torch.cat([self.up1_0(e2), e1], dim=1))

        d2_0 = self.conv2_0(torch.cat([self.up2_0(e3), e2], dim=1))
        d1_1 = self.conv1_1(torch.cat([self.up1_1(d2_0), e1, d1_0], dim=1))

        d3_0 = self.conv3_0(torch.cat([self.up3_0(e4), e3], dim=1))
        d2_1 = self.conv2_1(torch.cat([self.up2_1(d3_0), e2, d2_0], dim=1))
        d1_2 = self.conv1_2(torch.cat([self.up1_2(d2_1), e1, d1_0, d1_1], dim=1))

        d4_0 = self.conv4_0(torch.cat([self.up4_0(e5), e4], dim=1))
        d3_1 = self.conv3_1(torch.cat([self.up3_1(d4_0), e3, d3_0], dim=1))
        d2_2 = self.conv2_2(torch.cat([self.up2_2(d3_1), e2, d2_0, d2_1], dim=1))
        d1_3 = self.conv1_3(torch.cat([self.up1_3(d2_2), e1, d1_0, d1_1, d1_2], dim=1))

        return self.final(d1_3)
