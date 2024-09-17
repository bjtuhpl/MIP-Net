""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels1, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels1, out_channels2, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return  self.single_conv(x)

class SingleDown(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        return self.maxpool(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels1, out_channels2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class SingleUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x= torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels1, out_channels2, trilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels1, out_channels2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class TransConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, pre_channels, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(pre_channels, pre_channels,kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv(in_channels,out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class TransConv1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, pre_channels,in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(pre_channels, pre_channels,kernel_size=3, stride=2, padding=1)
        self.conv = SingleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class ConvDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv = DoubleConv(in_channels, in_channels, out_channels)
    def forward(self, x):
        x1 = self.conv_down(x)
        return  self.conv(x1)

class ConvDown1(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv0 = SingleConv(1, 12)
        # self.conv_out = nn.Sequential(
        #     nn.Conv3d(in_channels*2, in_channels, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = SingleConv(in_channels, out_channels)
    def forward(self, x,x0):
        x1 = self.conv_down(x)
        x0 = self.conv0(x0)

        # x1=torch.cat([x0,x1], dim=1)
        x1 = x0 + x1
        # x1 = self.conv_out(x1)
        return  self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        # # self.inc = DoubleConv(n_channels, 32, 64)
        # self.inc = SingleConv(n_channels, 8)
        # self.down1 = ConvDown1(8, 16)
        # self.down2 = ConvDown(16, 48)
        # self.down3 = ConvDown(48, 128)
        # self.down4 = ConvDown(128, 256)
        # self.down5 = ConvDown(256, 512)
        # self.up1 = TransConv(512,768, 256)
        # self.up2 = TransConv(256,384, 128)
        # self.up3 = TransConv(128,176, 48)
        # self.up4 = TransConv(48,64, 16)
        # self.up5 = TransConv1(16,24, 8)
        self.inc = SingleConv(n_channels, 12)
        self.down1 = ConvDown1(12, 24)
        self.down2 = ConvDown(24, 48)
        self.down3 = ConvDown(48, 128)
        self.down4 = ConvDown(128, 256)
        self.down5 = ConvDown(256, 512)
        self.up1 = TransConv(512,768, 256)
        self.up2 = TransConv(256,384, 128)
        self.up3 = TransConv(128,176, 48)
        self.up4 = TransConv(48,72,24)
        self.up5 = TransConv1(24,36, 12)
        # self.con= SingleConv(32, 32)

        self.outc = OutConv(12, n_classes)
        # self.conv = SingleConv(32, 32)

        # self.outc = OutConv(8, n_classes)

    def forward(self, x,x0):
        x1 = self.inc(x)
        x2 = self.down1(x1,x0)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        # x = self.conv(x)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0')
    x = torch.rand((1,1,128,128,128),device=device) # [bsize,channels,Height,Width,Depth]
    model = UNet(n_channels=1, n_classes=1)
    model.cuda(device)
    y = model(x)
    print(y.shape)