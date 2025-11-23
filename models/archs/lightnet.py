import torch
import torch.nn.functional as F
import numpy as np
from kornia.filters import sobel
from models.archs.DMRetinexLLIE import *
from models.archs.ffc import *
def pad_to(x, stride):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[2]+pad[3] > 0:
        x = x[:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,pad[0]:-pad[1]]
    return x

class Decoder(nn.Module):
    def __init__(self, channel, chan_factor,bias=False, n=2,groups=1):
        super(Decoder, self).__init__()
        self.up= nn.Sequential(
            nn.Conv2d(channel, int(channel // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
        )
        self.conv0 = nn.Conv2d(int(channel / chan_factor),int(channel / chan_factor),3,1,1,bias=bias)
        # self.conv0 = SpaBlock(int(n_feat / chan_factor))

        self.fusion= nn.Sequential(
            nn.Conv2d(2*int(channel / chan_factor),int(channel / chan_factor),3,1,1,bias=bias),
            FFCResnetBlock(int(channel / chan_factor)),
            # nn.BatchNorm2d(int(n_feat / chan_factor)),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x_in, x_en):
        x=self.conv0(self.up(x_in))
        x=self.fusion(torch.cat((x,x_en),dim=1))

        return x
class Encoder(nn.Module):
    def __init__(self, channel, chan_factor,bias=False):
        super(Encoder, self).__init__()

        # self.down=DownSample(n_feat, 2, chan_factor)
        self.down=nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(channel, channel * chan_factor, 1, stride=1, padding=0, bias=bias),
        )
        self.conv=nn.Sequential(
            #ProcessBlock(channel * chan_factor),
            FFCResnetBlock(channel * chan_factor),
        )

    def forward(self, x_in):
        x=self.down(x_in)
        x=self.conv(x)
        return x

class lightnessNet(nn.Module):
    def __init__(self,inp_channels=1,out_channels=1,channel=16,chan_factor=2,bias=False):

        super(lightnessNet, self).__init__()
        self.inp_channels=inp_channels
        self.out_channels=out_channels

        self.conv1 = nn.Conv2d(inp_channels, channel, kernel_size=3, padding=1, bias=bias)
        self.down1=Encoder(channel,chan_factor,bias)
        self.down2=Encoder(2 * channel,chan_factor,bias)
        self.down3=Encoder(4 * channel,chan_factor,bias)
        self.down4=Encoder(8 * channel,chan_factor,bias)

        self.up1 = Decoder(16 * channel, chan_factor, bias)
        self.up2 = Decoder(8 * channel, chan_factor, bias)
        self.up3 = Decoder(4 * channel, chan_factor, bias)
        self.up4 = Decoder(2 * channel, chan_factor, bias)

        self.conv2 = nn.Conv2d(channel, out_channels, kernel_size=3, padding=1, bias=bias)


    def forward(self, inp_img,l):
        inp_img, pads = pad_to(inp_img, 16)  # 测试时分辨率
        l, pads = pad_to(l, 16)
        inv_map = 1-inp_img  #注意力图？
        map=torch.sigmoid(inv_map)
        #edg_map = sobel(inp_img)
        x = self.conv1(l)
        down1=self.down1(x)
        down2=self.down2(down1)
        down3=self.down3(down2)
        bott=self.down4(down3)
        up1 = self.up1(bott,down3*map[:,:,::8,::8]+down3)
        up2 = self.up2(up1,down2*map[:,:,::4,::4]+down2)
        up3 = self.up3(up2,down1*map[:,:,::2,::2]+down1)
        up4 = self.up4(up3,x*map+x)
        out_img = self.conv2(up4)+l
        out_img = unpad(out_img, pads)
        return out_img


