import torch
from torch import nn

from models.encoders.crossnet import SEBlock
class FFM(nn.Module):
    def __init__(self,inc):
        super().__init__()
        self.conv_act=nn.Sequential(
            nn.Conv2d(2*inc,inc,3,1,1),
            nn.GELU()
        )
        self.ca=SEBlock('avg',inc,2)
        self.attforcross=SEBlock('avg',inc,2)
        self.attforvssm=SEBlock('avg',inc,2)
    def forward(self,x_vssm,x_cross):
        x=torch.cat([x_vssm,x_cross],dim=1)
        x=self.conv_act(x)
        x=self.ca(x)
        res_cross=self.attforcross(x_cross)
        res_vssm=self.attforvssm(x_vssm)
        out=x+res_vssm+res_cross
        return out

class SEM(nn.Module):
    def __init__(self):
        super(SEM,self).__init__()
        self.conv1=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=1)
        self.conv2=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=2,dilation=2)
        self.conv3=nn.Conv2d(2,1,kernel_size=3,stride=1,padding=3,dilation=3)
        self.act=nn.Sequential(
            nn.GroupNorm(1,1),
            nn.Hardsigmoid()
        )
    def forward(self,x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        d1=self.conv1(result)
        d2=self.conv2(result)
        d3=self.conv3(result)
        att=d1+d2+d3
        att=self.act(att)
        return x*att+x
class AFFM(nn.Module):
    def __init__(self,inc):
        super().__init__()
        self.conv_act=nn.Sequential(
            nn.Conv2d(2*inc,inc,3,1,1),
            nn.GELU()
        )
        self.ca=SEBlock('avg',inc,2)
        self.attforcross=nn.Sequential(
            SEM(),
            SEBlock('avg',inc,2),
        )
        self.attforvssm=nn.Sequential(
            SEM(),
            SEBlock('avg',inc,2),
        )
    def forward(self,x_vssm,x_cross):
        x=torch.cat([x_vssm,x_cross],dim=1)
        x=self.conv_act(x)
        x=self.ca(x)
        res_cross=self.attforcross(x_cross)
        res_vssm=self.attforvssm(x_vssm)
        out=x+res_vssm+res_cross
        return out
