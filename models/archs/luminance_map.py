import torch
import torch.nn as nn

class U_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(U_net, self).__init__()
        base_channel = 32
        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])
        # Middle
        self.middle = nn.Sequential(*[RB(base_channel * 8) for _ in range(depth[3])])
        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Up_scale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])
        # conv
        self.conv_first = BasicConv(1, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 1, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i // 3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, x):
        #x = x[ :, 0:1, :, :,]
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        #y = (torch.tanh(x) + 1) / 2
        y = torch.sigmoid(x)
        return y

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = nn.Conv2d(channels, channels, 3, 1)
        self.layer_2 = nn.Conv2d(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel * 2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel // 2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)