import functools
import torch.nn
import models.archs.arch_util as arch_util
from models.archs.luminance_map import *
from models.archs.onestage import *
import torch.nn.functional as F
from models.archs.twostage import EnhanceLayers
from models.archs.retinex import DecomNet
from utils.util import *

class DMRetinexLLIE(nn.Module):
    def __init__(self, nf=64):
        super(DMRetinexLLIE, self).__init__()
        self.phaseone = RetinexAugNet(nc=nf)
        self.sigmoid = torch.nn.Sigmoid()
        self.nf = nf
        self.twostage = EnhanceLayers(nf=nf)
        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, 1)
        self.cat = nn.Conv2d(nf * 2, nf, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=True)
        self.upconv1d = nn.Conv2d(nf, nf // 2, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        self.upconv2d = nn.Conv2d(nf // 2, nf // 8, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf // 4, nf // 8, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_last = nn.Conv2d(nf // 8, 3, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, low, fr_low, low_Y, normal, fr_normal):
        _, _, H, W = low.shape
        (pre_phaseone,
         R_fr_low, L_fr_low,
         R_fr_normal, L_fr_normal,
         R_low, L_low, L_low_aug,
         R_normal, L_normal) \
            = self.phaseone(low, fr_low, low_Y, normal, fr_normal)
        rate = 2 ** 3
        pad_h = (rate - H % rate) % rate
        pad_w = (rate - W % rate) % rate
        if pad_h != 0 or pad_w != 0:
            pre_phaseone = F.pad(pre_phaseone, (0, pad_w, 0, pad_h), "reflect")
            low = F.pad(low, (0, pad_w, 0, pad_h), "reflect")

        x1, x2, x3, f = self.twostage(torch.cat((pre_phaseone, low), dim=1))
        fea = f
        out_noise = self.recon_trunk(fea)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(torch.cat((out_noise, x3), dim=1))))
        out_noise = self.lrelu(self.upconv1d(out_noise))
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(torch.cat((out_noise, x2), dim=1))))
        out_noise = self.lrelu(self.upconv2d(out_noise))
        out_noise = self.lrelu(self.HRconv(torch.cat((out_noise, x1), dim=1)))
        out_noise = self.lrelu(self.conv_last(out_noise))
        out_noise = out_noise + low
        pre_phasetwo = out_noise[:, :, :H, :W]
        return (pre_phasetwo, pre_phaseone,
                R_fr_low, L_fr_low,
                R_fr_normal, L_fr_normal,
                R_low, L_low, L_low_aug,
                R_normal, L_normal)
