import torch
import torch.nn as nn
import torch.nn.functional as F

class CNR2d(nn.Module):
    def __init__(
        self,
        nch_in: int,
        nch_out: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
        norm: str = 'inorm',
        relu: float = 0.0,
        drop: float = None,
        bias: bool = None,
    ):
        super().__init__()
        # Normalize relu argument
        relu = None if relu == [] else relu
        if bias is None:
            bias = False if norm == 'inorm' else True
        layers = [Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if norm:
            layers.append(Norm2d(nch_out, norm))
        if relu is not None:
            layers.append(ReLU(relu))
        if drop is not None:
            layers.append(nn.Dropout2d(drop))
        self.cbr = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbr(x)

class DECNR2d(nn.Module):
    def __init__(
        self,
        nch_in: int,
        nch_out: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: int = 1,
        output_padding: int = 0,
        norm: str = 'inorm',
        relu: float = 0.0,
        drop: float = None,
        bias: bool = None,
    ):
        super().__init__()
        relu = None if relu == [] else relu
        if bias is None:
            bias = False if norm == 'inorm' else True
        layers = [Deconv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)]
        if norm:
            layers.append(Norm2d(nch_out, norm))
        if relu is not None:
            layers.append(ReLU(relu))
        if drop is not None:
            layers.append(nn.Dropout2d(drop))
        self.decbr = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decbr(x)

class ResBlock(nn.Module):
    def __init__(
        self,
        nch_in: int,
        nch_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: str = 'reflection',
        norm: str = 'inorm',
        relu: float = 0.0,
        drop: float = None,
        bias: bool = None,
    ):
        super().__init__()
        relu = None if relu == [] else relu
        if bias is None:
            bias = False if norm == 'inorm' else True
        layers = [
            Padding(padding, padding_mode=padding_mode),
            CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=relu),
        ]
        if drop is not None:
            layers.append(nn.Dropout2d(drop))
        layers += [
            Padding(padding, padding_mode=padding_mode),
            CNR2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=0, norm=norm, relu=None),
        ]
        self.resblk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.resblk(x)

class CNR1d(nn.Module):
    def __init__(
        self,
        nch_in: int,
        nch_out: int,
        norm: str = 'inorm',
        relu: float = 0.0,
        drop: float = None,
    ):
        super().__init__()
        relu = None if relu == [] else relu
        bias = False if norm == 'inorm' else True
        layers = [nn.Linear(nch_in, nch_out, bias=bias)]
        if norm:
            layers.append(Norm2d(nch_out, norm))
        if relu is not None:
            layers.append(ReLU(relu))
        if drop is not None:
            layers.append(nn.Dropout2d(drop))
        self.cbr = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbr(x)

class Conv2d(nn.Module):
    def __init__(self, nch_in: int, nch_out: int, kernel_size: int = 4, stride: int = 1, padding: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Deconv2d(nn.Module):
    def __init__(self, nch_in: int, nch_out: int, kernel_size: int = 4, stride: int = 1, padding: int = 1, output_padding: int = 0, bias: bool = True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(nch_in, nch_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)

class Linear(nn.Module):
    def __init__(self, nch_in: int, nch_out: int):
        super().__init__()
        self.linear = nn.Linear(nch_in, nch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class Norm2d(nn.Module):
    def __init__(self, nch: int, norm_mode: str):
        super().__init__()
        if norm_mode == 'inorm':
            self.norm = nn.InstanceNorm2d(nch, affine=True)
        elif norm_mode == 'bnorm':
            self.norm = nn.BatchNorm2d(nch)
        else:
            raise NotImplementedError(f"Normalization mode '{norm_mode}' is not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class ReLU(nn.Module):
    def __init__(self, relu):
        super().__init__()
        # Normalize relu argument
        if relu == [] or relu is None:
            self.relu = nn.ReLU(True)
        elif isinstance(relu, (float, int)) and relu > 0:
            self.relu = nn.LeakyReLU(relu, True)
        else:
            self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x)

class Padding(nn.Module):
    def __init__(self, padding: int, padding_mode: str = 'zeros', value: float = 0):
        super().__init__()
        if padding_mode == 'reflection':
            self.padding = nn.ReflectionPad2d(padding)
        elif padding_mode == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        elif padding_mode == 'constant':
            self.padding = nn.ConstantPad2d(padding, value)
        elif padding_mode == 'zeros':
            self.padding = nn.ZeroPad2d(padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.padding(x)

class Pooling2d(nn.Module):
    def __init__(self, nch: int = None, pool: int = 2, type: str = 'avg'):
        super().__init__()
        if type == 'avg':
            self.pooling = nn.AvgPool2d(pool)
        elif type == 'max':
            self.pooling = nn.MaxPool2d(pool)
        elif type == 'conv':
            self.pooling = nn.Conv2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(x)

class UnPooling2d(nn.Module):
    def __init__(self, nch: int = None, pool: int = 2, type: str = 'nearest'):
        super().__init__()
        if type == 'nearest':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='nearest', align_corners=True)
        elif type == 'bilinear':
            self.unpooling = nn.Upsample(scale_factor=pool, mode='bilinear', align_corners=True)
        elif type == 'conv':
            self.unpooling = nn.ConvTranspose2d(nch, nch, kernel_size=pool, stride=pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unpooling(x)

class Concat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        diffy = x2.size()[2] - x1.size()[2]
        diffx = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2])
        return torch.cat([x2, x1], dim=1)

class TV1dLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(input[:, :-1] - input[:, 1:]))

class TV2dLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        loss = torch.mean(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + \
               torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        return loss

class SSIM2dLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> float:
        # Placeholder for SSIM loss
        return 0.0

class GANLoss(nn.Module):
    """
    Define GAN loss for LSGAN (least squares GAN), as used in CycleGAN.
    """
    def __init__(self, use_lsgan: bool = True, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        return self.real_label.expand_as(input) if target_is_real else self.fake_label.expand_as(input)

    def forward(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

