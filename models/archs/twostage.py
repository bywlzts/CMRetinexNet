import torch
import torch.nn as nn
import torch.nn.functional as F
from models.archs.ffc import FFCResnetBlock

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class EnhanceBlock(nn.Module):
    def __init__(self, nf=64, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = nf * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=nf, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=nf, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.fftblock1 = FFCResnetBlock(nf)

        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )

        # GELU
        self.gelu = nn.GELU()

        ffn_channel = FFN_Expand * nf
        self.conv4 = nn.Conv2d(in_channels=nf, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=nf, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.norm1 = LayerNorm2d(nf)
        self.norm2 = LayerNorm2d(nf)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, nf, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, nf, 1, 1)), requires_grad=True)

    # def enhance(self, x):
    #     features1_flattened = x.view(vis.size(0), vis.size(1), -1)
    #
    #     multiplied = torch.mul(features1_flattened, features2_flattened)
    #     multiplied_softmax = torch.softmax(multiplied, dim=2)
    #     multiplied_softmax = multiplied_softmax.view(vis.size(0), vis.size(1), vis.size(2), vis.size(3))
    #     vis_map = vis * multiplied_softmax + vis
    #     return vis_map

    def forward(self, x3):

        inp = x3
        x4 = self.norm1(x3)
        x4 = self.fftblock1(x4)
        x4 = self.gelu(x4)
        x4 = x4 * self.se(x4)
        x4 = self.conv3(x4)
        x4 = self.dropout1(x4)

        y = inp + x4 * self.beta

        x5 = self.conv4(self.norm2(y))
        x5 = self.gelu(x5)
        # x=self.enhance(x)
        x5 = self.conv5(x5)

        x5 = self.dropout2(x5)

        return y + x5 * self.gamma


class EnhanceLayers(nn.Module):
    def __init__(self,nf=64):
        super().__init__()
        self.conv_first_1 = nn.Conv2d(3 * 2, nf // 8, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf // 8, nf // 2, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf // 2, nf, 3, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.en = nn.ModuleList()
        for i in range(4):
            layer = EnhanceBlock(nf=nf)
            self.en.append(layer)

    def forward(self, x):
        x1 = self.lrelu(self.conv_first_1(x))
        x2 = self.lrelu(self.conv_first_2(x1))
        x3 = self.lrelu(self.conv_first_3(x2))
        for m in self.en:
            f = m(x3)
        return x1,x2,x3,f
