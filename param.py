import torch
from torchvision.models import resnet18
from thop import profile
from models.archs.DMRetinexLLIE import DMRetinexLLIE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = DMRetinexLLIE(nf=16).to(device) 
input = torch.randn(1, 3, 256, 256).to(device) 
input2 = torch.randn(1, 3, 256, 256).to(device) 
input3 = torch.randn(1, 1, 256, 256).to(device) 
input4 = torch.randn(1, 3, 256, 256).to(device) 
input5 = torch.randn(1, 3, 256, 256).to(device) 
flops, params = profile(model, inputs=(input, input2, input3, input4, input5))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
