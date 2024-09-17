""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk



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

class ConvDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.sconv = SingleConv(in_channels, in_channels)
        self.conv0 = SingleConv(1, in_channels)
        # self.conv_out = nn.Sequential(
        #     nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = SingleConv(in_channels, out_channels)
    def forward(self, x,x0):
        x = self.conv_down(x)
        x1 = self.sconv(x)

        # diffY = torch.tensor([x.size()[2] - x0.size()[2]])
        # diffX = torch.tensor([x.size()[3] - x0.size()[3]])
        # diffZ = torch.tensor([x.size()[4] - x0.size()[4]])
        #
        # x0 = F.pad(x0, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2,
        #                 diffZ // 2, diffZ - diffZ // 2])
        x0 = self.conv0(x0)
        x1 = x1 + x0 + x
        m = nn.LayerNorm(x1.size()[1:]).cuda(torch.device('cuda:0'))
        x1 = m(x1)
        # x1 = torch.cat([x0, x], dim=1)
        # x1 = self.conv_out(x1)
        return self.conv(x1)

class SingleDown(nn.Module):
    def __init__(self, in_chanels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, x):
        return self.maxpool(x)

class DepthConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depth_conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.point_conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)

        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2,padding=1),
            DoubleConv(in_channels, out_channels1, out_channels2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Down1(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels1, out_channels2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2,padding=0)
        )
        self.sconv = SingleConv(in_channels, in_channels)
        self.conv0 = SingleConv(1, in_channels)
        # self.conv_out = nn.Sequential(
        #     nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, stride=1),
        #     nn.ReLU(inplace=True)
        # )
        self.conv = SingleConv(in_channels, out_channels2)

        # self.conv=DoubleConv(in_channels+1, out_channels1, out_channels2)

    def forward(self, x,x0):
        x = self.maxpool_conv(x)
        x1 = self.sconv(x)

        # diffY = torch.tensor([x.size()[2] - x0.size()[2]])
        # diffX = torch.tensor([x.size()[3] - x0.size()[3]])
        # diffZ = torch.tensor([x.size()[4] - x0.size()[4]])
        #
        # x0 = F.pad(x0, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2,
        #                 diffZ // 2, diffZ - diffZ // 2])
        x0 = self.conv0(x0)
        m = nn.LayerNorm(x1.size()[1:]).cuda(torch.device('cuda:0'))
        x1 = m(x1)
        # x1=torch.cat([x0,x], dim=1)
        # x1 = self.conv_out(x1)
        return self.conv(x1)

class SingleUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = SingleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffZ = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
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
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.convTrans = nn.ConvTranspose3d(in_channels, 1, kernel_size=3, stride=2,padding=0, output_padding=0,
                                            bias=True)
        self.conv = nn.Conv3d(2, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.convTrans(x)
        return x1
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=True):
        super(UNet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        #
        ## single channel input
        self.convdown = ConvDown(n_channels, 4)
        self.inc = SingleConv(4,6)
        self.down1 = Down1(6, 16, 16)
        self.down2 = Down(16, 64, 64)
        self.down3 = Down(64, 64, 128)
        self.down4 = Down(128, 128, 192)
        self.up0 = Up(320, 128, 128)
        self.up1 = Up(192, 64, 64, trilinear)
        self.up2 = Up(80, 16, 16, trilinear)
        self.up3 = SingleUp(22, 8, trilinear)
        # self.up4 = SingleUp(12, 4, trilinear)

        self.outc = OutConv(8, n_classes)

        # self.convdown = ConvDown(n_channels, 4)
        # self.inc = SingleConv(4,6)
        # self.down1 = Down1(6, 16, 16)
        # self.down2 = Down(16, 64, 64)
        # self.down3 = Down(64, 64, 128)
        # self.down4 = Down(128, 128, 192)
        # self.up0 = Up(320, 128, 128)
        # self.up1 = Up(192, 64, 64, trilinear)
        # self.up2 = Up(80, 16, 16, trilinear)
        # self.up3 = SingleUp(22, 4, trilinear)
        # # self.up4 = SingleUp(12, 4, trilinear)
        #
        # self.outc = OutConv(4, n_classes)

    def forward(self, x, first, second):

        x = self.convdown(x,first)
        x1 = self.inc(x)
        x2 = self.down1(x1,second)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up0(x5, x4)
        x = self.up1(x, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        # x = self.up4(x,)
        logits = self.outc(x)
        # for i in range(0,5):
        #     x_t = (x[:,i,:,:,:]).cpu().numpy().squeeze()
        #     gt = sitk.GetImageFromArray(x_t)
        #     sitk.WriteImage(gt, './data/debug/ConvPool_100_'+f'{i}' + '.nii.gz')
        return logits


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda:0')
    x = torch.rand((1,2,481,481,481),device=device)
    x0 = torch.rand((1, 1, 240, 240, 240), device=device)
    x1 = torch.rand((1, 1, 120, 120, 120), device=device)
    # [bsize,channels,Height,Width,Depth]
    model = UNet(n_channels=2, n_classes=1)
    model.cuda(device)
    y = model(x, x0, x1)
    print(y.shape)
    #9399