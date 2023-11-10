import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock

class ResnetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding="same"),
            nn.MaxPool2d((3,3), stride=2),
        )
        self.residual_blocks = nn.Sequential(
            BasicBlock(out_channels, out_channels),
            BasicBlock(out_channels, out_channels),
        )
        self.norm_out = nn.GroupNorm(1, out_channels)
    
    def forward(self, x):
        x = self.inp(x)

        x = self.residual_blocks(x)
        x = self.norm_out(x)
        return x

class BYOLEncoder(nn.Module):
    def __init__(self, in_channels, out_size, emb_dim=512):
        super().__init__()
        self.resnet_units = nn.Sequential(
            ResnetUnit(in_channels, 16),
            ResnetUnit(16, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32),
            ResnetUnit(32, 32)
        )

        self.out = nn.Linear(out_size, emb_dim)
    
    def forward(self, x):
        x = self.resnet_units(x)
        x = x.flatten(start_dim=1)
        
        x = self.out(x)
        return x


class ClosedLoopRNNCell(nn.Module):
    def __init__(self):
        super().__init__()
        pass

# class Temp(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(2100, 512),
#             nn.Linear(512, 512),
#             nn.Linear(512, 256),
#             nn.Linear(256, 8),
#         )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
net = Temp()

# encoder = BYOLEncoder(3, out_size=1344)
print(count_parameters(net))

# x = torch.randn((2, 3, 256, 240))
# print(encoder(x))