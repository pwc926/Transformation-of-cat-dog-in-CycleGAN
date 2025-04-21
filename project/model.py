from layer import *
import torch
import torch.nn as nn
from torch.nn import init

# UNet model definition
class UNet(nn.Module):
    def __init__(self, nch_in: int, nch_out: int, nch_ker: int = 64, norm: str = 'inorm'):
        super().__init__()
        bias = False if norm == 'inorm' else True
        self.enc1 = CNR2d(nch_in,  nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc2 = CNR2d(nch_ker, 2 * nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc3 = CNR2d(2 * nch_ker, 4 * nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc4 = CNR2d(4 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc5 = CNR2d(8 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc6 = CNR2d(8 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc7 = CNR2d(8 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.2)
        self.enc8 = CNR2d(8 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.0)
        self.dec8 = DECNR2d(8 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.0, drop=0.5)
        self.dec7 = DECNR2d(16 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.0, drop=0.5)
        self.dec6 = DECNR2d(16 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.0, drop=0.5)
        self.dec5 = DECNR2d(16 * nch_ker, 8 * nch_ker, stride=2, norm=norm, relu=0.0)
        self.dec4 = DECNR2d(16 * nch_ker, 4 * nch_ker, stride=2, norm=norm, relu=0.0)
        self.dec3 = DECNR2d(8 * nch_ker, 2 * nch_ker, stride=2, norm=norm, relu=0.0)
        self.dec2 = DECNR2d(4 * nch_ker, nch_ker, stride=2, norm=norm, relu=0.0)
        self.dec1 = DECNR2d(2 * nch_ker, nch_out, stride=2, norm=[], relu=[], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)
        d8 = self.dec8(e8)
        d7 = self.dec7(torch.cat([e7, d8], 1))
        d6 = self.dec6(torch.cat([e6, d7], 1))
        d5 = self.dec5(torch.cat([e5, d6], 1))
        d4 = self.dec4(torch.cat([e4, d5], 1))
        d3 = self.dec3(torch.cat([e3, d4], 1))
        d2 = self.dec2(torch.cat([e2, d3], 1))
        d1 = self.dec1(torch.cat([e1, d2], 1))
        return torch.tanh(d1)

# ResNet model definition
class ResNet(nn.Module):
    def __init__(self, nch_in: int, nch_out: int, nch_ker: int = 64, norm: str = 'inorm', n_blocks: int = 9):
        super().__init__()
        layers = [
            CNR2d(nch_in, nch_ker, kernel_size=7, stride=1, padding=3, norm=norm, relu=0.0),
            CNR2d(nch_ker, 2 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0),
            CNR2d(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0),
        ]
        layers += [
            ResBlock(4 * nch_ker, 4 * nch_ker, kernel_size=3, stride=1, padding=1, padding_mode='reflection', norm=norm, relu=0.0)
            for _ in range(n_blocks)
        ]
        layers += [
            DECNR2d(4 * nch_ker, 2 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0),
            DECNR2d(2 * nch_ker, nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.0),
            CNR2d(nch_ker, nch_out, kernel_size=7, stride=1, padding=3, norm=[], relu=[], bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.model(x))

# Discriminator model definition
class Discriminator(nn.Module):
    def __init__(self, nch_in: int, nch_ker: int = 64, norm: str = 'inorm'):
        super().__init__()
        layers = [
            CNR2d(nch_in, nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2),
            CNR2d(nch_ker, 2 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2),
            CNR2d(2 * nch_ker, 4 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2),
            CNR2d(4 * nch_ker, 8 * nch_ker, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2),
            CNR2d(8 * nch_ker, 1, kernel_size=4, stride=1, padding=1, norm=[], relu=[], bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Weight initialization functions
def init_weights(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)

def init_net(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02, gpu_ids=[]):
    if gpu_ids:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        net = nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net

