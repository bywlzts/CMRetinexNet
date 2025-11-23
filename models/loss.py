import torch
import torch.nn as nn
from utils.util import *
import cv2
import torchvision.transforms as transforms


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss


##############
class CharbonnierLoss2(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss2, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


import torchvision
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        # self.criterion = nn.L1Loss()
        self.criterion = nn.L1Loss(reduction='sum')
        self.criterion2 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward2(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            # print(x_vgg[i].shape, y_vgg[i].shape)
            loss += self.weights[i] * self.criterion2(x_vgg[i], y_vgg[i].detach())
        return loss


import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, inpput, gt):
        style_loss = self.l1(gram_matrix(inpput),
                             gram_matrix(gt))
        return style_loss


class ComposeLoss(nn.Module):
    def __init__(self):
        super(ComposeLoss, self).__init__()
        self.l1=nn.L1Loss()

    def forward(self, R_low, R_high,I_low,I_high,input_low,input_high):
        I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
        I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)
        #print(R_low.size(), I_low_3.size())
        recon_loss_low = self.l1(R_low * I_low_3, input_low)
        Ismooth_loss_low = self.smooth(I_low, R_low)
        Ismooth_loss_high = self.smooth(I_high, R_high)

        recon_loss_high = self.l1(R_high * I_high_3, input_high)
        recon_loss_mutal_low = self.l1(R_high * I_low_3, input_low)
        recon_loss_mutal_high = self.l1(R_low * I_high_3, input_high)
        equal_R_loss = self.l1(R_low, R_high.detach())
        loss1 = recon_loss_low + \
                          recon_loss_high + \
                          0.001 * recon_loss_mutal_low + \
                          0.001 * recon_loss_mutal_high + \
                          0.1 * Ismooth_loss_low + \
                          0.1 * Ismooth_loss_high + \
                          0.01 * equal_R_loss
        return loss1

    def gradient(self, input_tensor, direction):
        self.smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda()
        self.smooth_kernel_y = torch.transpose(self.smooth_kernel_x, 2, 3)

        if direction == "x":
            kernel = self.smooth_kernel_x
        elif direction == "y":
            kernel = self.smooth_kernel_y
        grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                      stride=1, padding=1))
        return grad_out

    def ave_gradient(self, input_tensor, direction):
        return F.avg_pool2d(self.gradient(input_tensor, direction),
                            kernel_size=3, stride=1, padding=1)

    def smooth(self, input_I, input_R):
        input_R = 0.299*input_R[:, 0, :, :] + 0.587*input_R[:, 1, :, :] + 0.114*input_R[:, 2, :, :]
        input_R = torch.unsqueeze(input_R, dim=1)
        return torch.mean(self.gradient(input_I, "x") * torch.exp(-10 * self.ave_gradient(input_R, "x")) +
                          self.gradient(input_I, "y") * torch.exp(-10 * self.ave_gradient(input_R, "y")))

# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.sobelconv=Sobelxy()
#
#     def forward(self,image_vis,image_ir,generate_img):
#         image_y = image_vis
#         x_in_max = torch.max(image_y, image_ir)
#         loss_in = F.l1_loss(x_in_max, generate_img)
#         y_grad = self.sobelconv(image_y)
#         ir_grad = self.sobelconv(image_ir)
#         generate_img_grad = self.sobelconv(generate_img)
#         x_grad_joint = torch.max(y_grad, ir_grad)
#         loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)
#         loss_total = loss_in + 10 * loss_grad
#         return loss_total
#
# class Sobelxy(nn.Module):
#     def __init__(self):
#         super(Sobelxy, self).__init__()
#         kernelx = [[-1, 0, 1],
#                   [-2,0 , 2],
#                   [-1, 0, 1]]
#         kernely = [[1, 2, 1],
#                   [0,0 , 0],
#                   [-1, -2, -1]]
#         kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
#         kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
#         self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
#         self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
#     def forward(self,x):
#         sobelx=F.conv2d(x, self.weightx, padding=1)
#         sobely=F.conv2d(x, self.weighty, padding=1)
#         return torch.abs(sobelx)+torch.abs(sobely)
class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        Loss_SSIM = 0.5 * ssim(image_A, image_fused) + 0.5 * ssim(image_B, image_fused)
        return Loss_SSIM

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        image_A_Y = image_A[:, :1, :, :]
        image_B_Y = image_B[:, :1, :, :]
        image_fused_Y = image_fused[:, :1, :, :]
        gradient_A = self.sobelconv(image_A_Y)
        # gradient_A = TF.gaussian_blur(gradient_A, 3, [1, 1])
        gradient_B = self.sobelconv(image_B_Y)
        # gradient_B = TF.gaussian_blur(gradient_B, 3, [1, 1])
        gradient_fused = self.sobelconv(image_fused_Y)
        gradient_joint = torch.max(gradient_A, gradient_B)
        Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return Loss_gradient

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

# class L_Intensity(nn.Module):
#     def __init__(self):
#         super(L_Intensity, self).__init__()
#
#     def forward(self, image_A, image_B, image_fused):
#         intensity_joint = torch.max(image_A, image_B)
#         Loss_intensity = F.l1_loss(image_fused, intensity_joint)
#         return Loss_intensity
#
# class Fusionloss(nn.Module):
#     def __init__(self):
#         super(Fusionloss, self).__init__()
#         self.L_Grad = L_Grad()
#         self.L_Inten = L_Intensity()
#         self.L_SSIM = L_SSIM()
#
#         # print(1)
#     def forward(self, image_A, image_B, image_fused):
#         # image_A represents MRI image
#         loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
#         loss_gradient = 100 * self.L_Grad(image_A, image_B, image_fused)
#         loss_SSIM = 50 * (1 - self.L_SSIM(image_A, image_B, image_fused))
#         fusion_loss = loss_l1 + loss_gradient + loss_SSIM
#         return fusion_loss

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

    def rgbtoy(self, x):
        R=x[:,:1,:,:]
        G=x[:,1:2,:,:]
        B=x[:,2:3,:,:]
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128 / 255
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128 / 255


        return Y, Cb, Cr

        # print(1)
    def forward(self, image_A, image_B, image_fused, R, fusion):
        # image_A represents MRI image
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 100 * self.L_Grad(image_A, image_B, image_fused)
        loss_SSIM = 50 * (1 - self.L_SSIM(image_A, image_B, image_fused))
        # color_angle_loss = 50*torch.mean(
        #      self.angle(fusion[:, :, :, 0], R[:, :, :, 0]) + self.angle(fusion[:, :, :, 1], R[:, :, :, 1]) + self.angle(fusion[:, :, :, 2],R[:, :, :, 2]))
        fusion_loss = loss_l1+loss_gradient+loss_SSIM
        return fusion_loss

    # def angle(self, a, b):
    #     #     vector = torch.multiply(a, b)
    #     #     up = torch.sum(vector)
    #     #     v=1e-7
    #     #     down = torch.sqrt(torch.sum(torch.square(a))+v) * torch.sqrt(torch.sum(torch.square(b))+v)
    #     #     theta = torch.acos(up / (down+v))  # ������
    #     #     return theta
    def angle(self, a, b):
        vector = torch.multiply(a, b)
        up = torch.sum(vector)
        v = 1e-7
        down = torch.sqrt(torch.sum(torch.square(a)) + v) * torch.sqrt(torch.sum(torch.square(b)) + v)
        x1 = up / (down + v)
        x1 = torch.clip(x1, -1+v , 1+v )
        theta = torch.acos(x1)  # ??????
        return theta


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

# class lightloss(nn.Module):
#     """Charbonnier Loss (L1)"""
#
#     def __init__(self, eps=1e-6):
#         super(lightloss, self).__init__()
#         self.tvloss=TVLoss()
#         self.Charbonnier=CharbonnierLoss2()
#
#     def forward(self, x, y):
#         tv=self.tvloss(x)
#         Charbonnier=self.Charbonnier(x,y)
#         ssim1=1-ssim(x,y)
#         loss=tv+Charbonnier+0.1*ssim1
#         return loss

class lightloss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self):
        super(lightloss, self).__init__()
        self.l2=nn.MSELoss()

    def forward(self, x, y):
        loss=self.l2(x,y)
        return loss
        
class FourierAmplitudeLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super(FourierAmplitudeLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid loss_type. Use 'l1' or 'l2'.")

    def forward(self, input, target):
        # 计算输入和目标图像的傅里叶变换
        fft_input = torch.fft.fftn(input, dim=(-2, -1))
        fft_target = torch.fft.fftn(target, dim=(-2, -1))

        # 计算傅里叶频谱的振幅
        mag_input = torch.abs(fft_input)
        mag_target = torch.abs(fft_target)

        # 计算傅里叶振幅损失
        loss = self.loss(mag_input, mag_target)
        return loss


class FourierPhaseLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super(FourierPhaseLoss, self).__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Invalid loss_type. Use 'l1' or 'l2'.")

    def forward(self, input, target):
        # 计算输入和目标图像的傅里叶变换
        fft_input = torch.fft.fftn(input, dim=(-2, -1))
        fft_target = torch.fft.fftn(target, dim=(-2, -1))

        # 计算傅里叶频谱的相位
        phase_input = torch.angle(fft_input)
        phase_target = torch.angle(fft_target)

        # 计算傅里叶相位损失
        loss = self.loss(phase_input, phase_target)
        return loss


class CombinedFourierLoss(nn.Module):
    def __init__(self, amplitude_weight=1.0, phase_weight=1.0, loss_type='l1'):
        super(CombinedFourierLoss, self).__init__()
        self.amplitude_weight = amplitude_weight
        self.phase_weight = phase_weight
        self.amplitude_loss = FourierAmplitudeLoss(loss_type)
        self.phase_loss = FourierPhaseLoss(loss_type)

    def forward(self, input, target):
        # 计算傅里叶振幅损失
        amplitude_loss = self.amplitude_loss(input, target)

        # 计算傅里叶相位损失
        phase_loss = self.phase_loss(input, target)

        # 组合损失
        total_loss = self.amplitude_weight * amplitude_loss + self.phase_weight * phase_loss
        return total_loss