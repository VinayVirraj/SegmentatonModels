import torch
import torch.nn as nn


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        _, _, current_height, current_width = x2.shape
        _, _, target_height, target_width = x1.shape
        crop_height = (current_height - target_height) // 2
        crop_width = (current_width - target_width) // 2
        cropped_output = x2[:, :,crop_height:crop_height + target_height, crop_width:crop_width + target_width]

        x = torch.cat([cropped_output, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, configs):
        super(UNet, self).__init__()
        self.in_channels = configs['in_channels']
        self.out_channels = configs['out_channels']

        self.enc1 = (DoubleConvBlock(self.in_channels, 64))
        self.enc2 = (DownConvBlock(64, 128))
        self.enc3 = (DownConvBlock(128, 256))
        self.enc4 = (DownConvBlock(256, 512))
        self.bottleneck = (DownConvBlock(512, 1024))
        self.dec4 = (UpConvBlock(1024, 512))
        self.dec3 = (UpConvBlock(512, 256))
        self.dec2 = (UpConvBlock(256, 128))
        self.dec1 = (UpConvBlock(128, 64))
        self.outc = (OutConv(64, self.out_channels))

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)
        x = self.dec4(x5,x4)
        x = self.dec3(x,x3)
        x = self.dec2(x,x2)
        x = self.dec1(x,x1)
        out = self.outc(x)
        return torch.sigmoid(out)