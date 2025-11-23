import torch.nn as nn
from utils.util import *
from models.archs.retinex import DecomNet
from models.archs.lightnet import lightnessNet

class RetinexAugUnit(nn.Module):
    def __init__(self, nc):
        super(RetinexAugUnit, self).__init__()
        self.fpre = nn.Conv2d(nc, nc, 1, 1, 0)
        self.catconv = nn.Conv2d(nc*2, nc, 1,1,0)
        #self.fm2 = MambaDFuse()
        self.lightnet = lightnessNet()
        self.decomnet = DecomNet(channel=nc)
        self.process_out = nn.Sequential(
            nn.Conv2d(3, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.process_sp = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.combine_net = CombineNet()

    def forward(self, fpre_low, fr_low, Y_map, normal, low,fr_normal):
        _, _, H, W = fpre_low.shape

        low_spatial = self.process_sp(fpre_low)
        low_spatial = fpre_low + low_spatial

        R_fr_low, L_fr_low = self.decomnet(fr_low)
        R_fr_normal, L_fr_normal = self.decomnet(fr_normal)

        R_low, L_low = self.decomnet(low)
        R_normal, L_normal = self.decomnet(normal)

        L_low_aug = self.lightnet(Y_map, L_low)

        pre_retinex = self.combine_net(R_low + L_low_aug)
        pre_retinex = self.process_out(pre_retinex)

        #pre_phaseone = self.catconv(torch.cat([pre_retinex, low_spatial],1)) + fpre_low
        return (pre_retinex,
                R_fr_low, L_fr_low,
                R_fr_normal, L_fr_normal,
                R_low, L_low, L_low_aug,
                R_normal, L_normal)

class RetinexAugNet(nn.Module):
    def __init__(self, nc):
        super(RetinexAugNet, self).__init__()
        self.conv0a = nn.Conv2d(3, nc, 1, 1, 0)
        self.augunit = RetinexAugUnit(nc)
        self.last = nn.Conv2d(nc, 3, 1, 1, 0)

    def forward(self, low, fr, y_map, normal, fr_normal):
        pre_phaseone, R_fr_low, L_fr_low, R_fr_normal, L_fr_normal, R_low, L_low, L_low_aug, R_normal, L_normal = self.augunit(self.conv0a(low), fr, y_map, normal, low, fr_normal)
        pre_phaseone = self.last(pre_phaseone)
        return pre_phaseone, R_fr_low, L_fr_low, R_fr_normal, L_fr_normal, R_low, L_low, L_low_aug, R_normal, L_normal


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class CombineNet(nn.Module):
    def __init__(self):
        super(CombineNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.ca1 = ChannelAttention(16)
        #self.sa1 = SpatialAttention()

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.ca2 = ChannelAttention(16)
        #self.sa2 = SpatialAttention()

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.ca3 = ChannelAttention(16)
        #self.sa3 = SpatialAttention()

        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.relu4 = nn.ReLU()
        # self.ca4 = ChannelAttention(64)
        # #self.sa4 = SpatialAttention()

        # self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # self.relu5 = nn.ReLU()
        # self.ca5 = ChannelAttention(64)
        # #self.sa5 = SpatialAttention()

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()

        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, x):
        residual0 = x
        x = self.conv1(x)
        residual = x
        x = self.relu1(x)
        x = self.ca1(x)


        x = self.conv2(x) + residual
        residual = x
        x = self.relu2(x)
        x = self.ca2(x)
        #attention = self.sa2(x)
        #x = x * attention
        #residual = x

        x = self.conv3(x) + residual
        residual = x
        x = self.relu3(x)
        x = self.ca3(x)
        #attention = self.sa3(x)
        #x = x * attention
        #residual = x

        # x = self.conv4(x) + residual
        # x = self.relu4(x)
        # x = self.ca4(x)
        # #attention = self.sa4(x)
        # #x = x * attention
        # #residual = x

        # x = self.conv5(x) + residual
        # x = self.relu5(x)
        # x = self.ca5(x)
        #attention = self.sa5(x)
        #x = x * attention
        #residual = x

        x = self.conv6(x) + residual
        x = self.relu6(x)

        x = self.final_conv(x) + residual0

        return x