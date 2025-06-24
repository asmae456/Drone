import torch.nn as nn
from tools import QLinear
from tools import  Normalize
import torch
import torch.nn as nn


class Drone(nn.Module):
    def __init__(self, wbits=4, abits=4, X_mean=0, X_std=0):
        super(Drone, self).__init__()
        self.abits = abits
        self.wbits = wbits

        self.X_mean = X_mean
        self.X_std = X_std

        if self.abits == 32:
            act = nn.ReLU(inplace=True)
        else:
            # act = ScaledReLU(scale=1/16)
            act = nn.ReLU(inplace=True)
            act1 = CustomSigmoid2(abits=self.abits, wbits=self.wbits)
            
        
        # Build model using nn.Sequential so it matches saved structure
        self.model = nn.Sequential(
            Normalize(mean=self.X_mean, std=self.X_std),
            QLinear(abits=self.abits, wbits=self.wbits, in_features=16, out_features=120, bias=True),
            act,
            QLinear(abits=self.abits, wbits=self.wbits, in_features=120, out_features=120, bias=True),
            act,
            QLinear(abits=self.abits, wbits=self.wbits, in_features=120, out_features=120, bias=True),
            act,
            QLinear(abits=self.abits, wbits=self.wbits, in_features=120, out_features=4, bias=True),
            act1
        )

    def forward(self, x):
        return self.model(x)
class CustomSigmoid2(nn.Module):
    def __init__(self, abits=4, wbits=4):
        super().__init__()
        self.abits = abits
        self.wbits = wbits

    def forward(self, x):
        x_abs = torch.abs(x)
        # print(torch.max(x))
        y =  ( self.abits*self.wbits)/2 + (x_abs) * 1/128
        # y = y * 2**5
        # Mirror negative values using sign
        y = torch.where(x >= 0, y, ( self.abits*self.wbits) - y)
        # y = x
        y = torch.clamp(y, 0.0,(self.abits*self.wbits))
        y = y/(self.abits*self.wbits)
        return y   
class CustomSigmoid3(nn.Module):
    def __init__(self, abits=4, wbits=4):
        super().__init__()
        self.abits = abits
        self.wbits = wbits

    def forward(self, x):
        x_abs = torch.abs(x)
        print(torch.max(x))
        # print(x_abs)(x_abs.to(torch.int32) >> 5)  (x_abs * (1/32)
        y =  (2 ** (self.abits-1)) + (x_abs * 0.03 )
        y = y * 2**5
        # Mirror negative values using sign
        y = torch.where(x >= 0, y, (2**5 * self.abits*self.wbits) - y)
        # y = x
        y = torch.clamp(y, 0.0,(2**5 * self.abits*self.wbits))
        y = y/(2**5 * self.abits*self.wbits)
        return y
class ScaledReLU(nn.Module):
    def __init__(self, scale=1/16):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        return self.relu(x) * self.scale
class CustomSigmoid1(nn.Module):
    def forward(self, x):
        x_abs = torch.abs(x)
        y = (1/2) + (1/4)*x_abs + (1/8)*x_abs + (1/64)*x_abs - (1/256)*(x_abs**2)
        
        # Mirror negative values using sign
        y = torch.where(x >= 0, y, 1 - y)

        return torch.clamp(y, 0.0, 1.0)

    
class CustomSigmoid(nn.Module):
    def forward(self, x):
        return   torch.clamp(0.5 + 0.197 * x - 0.004 * x**3, 0.0, 1.0)

# class CustomSigmoid1(nn.Module):
#     def forward(self, x):
#         x_abs = torch.abs(x)
#         y = (1/2) + (1/4)*x_abs + (1/8)*x_abs + (1/64)*x_abs - (1/256)*(x_abs**3)
        
#         # Mirror negative values using sign
#         y = torch.where(x >= 0, y, 1 - y)

#         return torch.clamp(y, 0.0, 1.0)
     
class PiecewiseSigmoid(nn.Module):
    def forward(self, x):
        return torch.where(x < -2.5, torch.zeros_like(x),
               torch.where(x > 2.5, torch.ones_like(x),
               0.2 * x + 0.5))
class TanhSigmoid(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x / 2) + 1)
    

class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)

class FastSigmoid(nn.Module):
    def forward(self, x):
        return x / (1 + torch.abs(x))

