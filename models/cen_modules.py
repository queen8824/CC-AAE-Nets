import torch.nn as nn
import torch
import torch.nn.functional as F

class Exchange(nn.Module): #通道交换关键代码
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ConvParallel(nn.Module):
    def __init__(self, module1,module2):
        super(ConvParallel, self).__init__()
        self.module1 = module1
        self.module2 = module2

    def forward(self, x_parallel):
        result = []
        r1 = self.module1(x_parallel[0])
        result.append(r1)
        r2 = self.module2(x_parallel[1])
        result.append(r2)
        return result


class AvgParallel(nn.Module):
    def __init__(self, module):
        super(AvgParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x).view(x.size(0), -1) for x in x_parallel]

class AdaptiveAvgParallel(nn.Module):
    def __init__(self, module):
        super(AdaptiveAvgParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        result = []
        b,c = 0,0
        for x in x_parallel:
            b, c, _, _ = x.size()
            y = self.module(x).view(b, c)
            result.append(y)
        return result,b,c

# class MlpParallel(nn.Module):
#     def __init__(self, module):
#         super(MlpParallel, self).__init__()
#         self.module = module
#
#     def forward(self, x_parallel,y_parrallel):
#         return [self.module(x).view(x.size(0), -1) for x in x_parallel]
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class BatchNorm2dParallel(nn.Module):
    def __init__(self, num_features, num_parallel=2):
        super(BatchNorm2dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features))
            # setattr(self, 'bn_' + str(i), LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class BatchNorm2dParallelB(nn.Module):
    def __init__(self, num_features, momentum, eps, num_parallel=2):
        super(BatchNorm2dParallelB, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm2d(num_features,momentum,eps))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]
