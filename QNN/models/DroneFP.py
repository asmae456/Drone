import torch
import torch.nn as nn
from tools import Normalize  # Keep Normalize if it's needed for input preprocessing

class DroneFP(nn.Module):
    def __init__(self, X_mean=0, X_std=0):
        super(DroneFP, self).__init__()

        self.model = nn.Sequential(
            Normalize(mean=X_mean, std=X_std),
            nn.Linear(in_features=16, out_features=120, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=120, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=120, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=4, bias=True),
            CustomSigmoid2()
        )

    def forward(self, x):
        return self.model(x)

class CustomSigmoid(nn.Module):
    def forward(self, x):
        return   torch.clamp(0.5 + 0.197 * x - 0.004 * x**3, 0.0, 1.0)

class CustomSigmoid1(nn.Module):
    def forward(self, x):
        x_abs = torch.abs(x)
        y = (1/2) + (1/4)*x_abs + (1/8)*x_abs + (1/64)*x_abs - (1/256)*(x_abs**3)
        
        # Mirror negative values using sign
        y = torch.where(x >= 0, y, 1 - y)

        return torch.clamp(y, 0.0, 1.0)

class PiecewiseSigmoid(nn.Module):
    def forward(self, x):
        return torch.where(x < -2.5, torch.zeros_like(x),
               torch.where(x > 2.5, torch.ones_like(x),
               0.2 * x + 0.5))
class TanhSigmoid(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x / 2) + 1)

class FastSigmoid(nn.Module):
    def forward(self, x):
        return x / (1 + torch.abs(x))
class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)
class CustomSigmoid2(nn.Module):
    def forward(self, x):
        x_abs = torch.abs(x)
        # print(x)
        y = (1/2) + (1/32) * x_abs 
        #+ (2**(-11)/(64))*(x_abs**2)
        
        # Mirror negative values using sign
        y = torch.where(x >= 0, y, 1 - y)
        # y = x
        return torch.clamp(y, 0.0, 1.0)