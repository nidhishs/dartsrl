import torch
import torch.nn as nn
import torch.nn.functional as F

OPERATIONS = {
    'none': lambda c, stride, affine: Zero(stride),
    'noise': lambda C, stride, affine: NoiseOp(stride, 0.0, 1.0),
    'avg_pool_3x3': lambda c, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda c, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda c, stride, affine: nn.Identity() if stride == 1 else FactorizedReduce(c, c, affine=affine),
    'sep_conv_3x3': lambda c, stride, affine: SepConv(c, c, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda c, stride, affine: SepConv(c, c, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda c, stride, affine: SepConv(c, c, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda c, stride, affine: DilConv(c, c, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda c, stride, affine: DilConv(c, c, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda c, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(c, c, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(c, affine=affine)
    ),
    'conv_1x1': lambda c, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c, 1, stride=stride, padding=0, bias=False),
        nn.BatchNorm2d(c, affine=affine)
    ),
    'conv_3x3': lambda c, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(c, c, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(c, affine=affine)
    ),
}


SEARCH_SPACES = {
    'darts': [
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5',
        'skip_connect',
        'avg_pool_3x3',
        'max_pool_3x3',
        'none'
    ],
    's2': [
        'skip_connect',
        'sep_conv_3x3',
    ],
    's3': [
        'none',
        'skip_connect',
        'sep_conv_3x3',
    ],
    's4': [
        'noise',
        'sep_conv_3x3',
    ],
    'nas-bench-201': [
        'none',
        'conv_1x1',
        'conv_3x3',
        'avg_pool_3x3',
        'skip_connect',
    ]
}


class DilConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(c_in, c_in, kernel_size, stride, padding, dilation=dilation, groups=c_in, bias=False),
            nn.Conv2d(c_in, c_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class SepConv(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(c_in, c_in, kernel_size, stride, padding, dilation=1, affine=affine),
            DilConv(c_in, c_out, kernel_size, 1, padding, dilation=1, affine=affine)
        )

    def forward(self, x):
        return self.net(x)


class FactorizedReduce(nn.Module):
    def __init__(self, c_in, c_out, affine=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(c_in, c_out//2, 1, stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(c_in, c_out//2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Zero(nn.Module):
    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.

        return x[:, :, ::self.stride, ::self.stride] * 0.


class ReLUConvBN(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride, padding, affine=True):
        super().__init__()
        self.op = nn.Sequential(
          nn.ReLU(inplace=False),
          nn.Conv2d(c_in, c_out, kernel_size, stride=stride, padding=padding, bias=False),
          nn.BatchNorm2d(c_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)
    

class NoiseOp(nn.Module):
    def __init__(self, stride, mean, std):
        super().__init__()
        self.stride = stride
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.stride != 1:
            x_new = x[:, :, ::self.stride, ::self.stride]
        else:
            x_new = x
        noise = torch.randn_like(x_new) * self.std + self.mean
        return noise


class MixedOp(nn.Module):
    def __init__(self, c, stride, primitives, dropout_proba):
        super().__init__()
        self._ops = nn.ModuleList()
        
        for primitive in primitives:
            op = OPERATIONS[primitive](c, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(c, affine=False))
            if isinstance(op, nn.Identity) and dropout_proba > 0:
                op = nn.Sequential(op, nn.Dropout(dropout_proba))
            self._ops.append(op)

    def forward(self, x, weights, eps=1e-3):
        return sum([w * op(x) for w, op in zip(weights, self._ops) if w > eps])
