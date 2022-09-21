"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from utils.helpers import maybe_download
from .cen_modules import *

data_info = {
    21: 'VOC',
}

models_urls = {
    '101_voc': 'https://cloudstor.aarnet.edu.au/plus/s/Owmttk9bdPROwc6/download',
    '18_imagenet': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34_imagenet': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50_imagenet': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101_imagenet': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152_imagenet': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold  # for silmming bns
        self.bn1_list = []
        for module in self.bn1.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn1_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn1_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        # self.exchange = Exchange()
        self.bn_threshold = bn_threshold  # for silmming bns
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # if len(x) > 1:
        #     out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1,num_parallel=2):
        super(ChannelGate, self).__init__()
        # self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', ModuleParallel(Flatten()) )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, ModuleParallel(nn.Linear(gate_channels[i], gate_channels[i+1])) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), ModuleParallel(nn.BatchNorm1d(gate_channels[i+1])) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), ModuleParallel(nn.ReLU()) )
        self.gate_c.add_module( 'gate_c_fc_final', ModuleParallel(nn.Linear(gate_channels[-2], gate_channels[-1])) )
        self.num_parallel = num_parallel
    def forward(self, in_tensor):
        avg_pool = [ F.avg_pool2d( in_tensor[x], in_tensor[x].size(2), stride=in_tensor[x].size(2)) for x in  range(self.num_parallel) ]
        gate_c = self.gate_c( avg_pool )
        return [ gate_c[x].unsqueeze(2).unsqueeze(3).expand_as(in_tensor[x]) for x in  range(self.num_parallel) ]

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4,num_parallel=2):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', ModuleParallel(nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1)))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	ModuleParallel(nn.BatchNorm2d(gate_channel//reduction_ratio)) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',ModuleParallel(nn.ReLU()) )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, ModuleParallel(nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val)) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, ModuleParallel(nn.BatchNorm2d(gate_channel//reduction_ratio)) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, ModuleParallel(nn.ReLU()) )
        self.gate_s.add_module( 'gate_s_conv_final', ModuleParallel(nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1)) )
        self.num_parallel = num_parallel
    def forward(self, in_tensor):
        gate_s = self.gate_s( in_tensor )
        return [ gate_s[x].expand_as(in_tensor[x]) for x in range(self.num_parallel) ]
class BAM(nn.Module):
    def __init__(self, gate_channel,num_parallel):
        super(BAM, self).__init__()
        self.num_parallel = num_parallel
        self.channel_att = ChannelGate(gate_channel,self.num_parallel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = [ 1 + F.sigmoid( self.channel_att(in_tensor)[x] * self.spatial_att(in_tensor)[x] ) for x in range(self.num_parallel) ]
        return [ att[x] * in_tensor[x] for x in range(self.num_parallel) ]
class RefineNet(nn.Module):
    def __init__(self, block, layers, num_parallel, num_classes=1, bn_threshold=2e-2):
        self.inplanes = 64
        self.num_parallel = num_parallel  # 2
        super(RefineNet, self).__init__()
        self.dropout = ModuleParallel(nn.Dropout(p=0.5))
        self.conv1 = ModuleParallel(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False))
        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.bam1 = BAM(64 * block.expansion,self.num_parallel)
        self.bam2 = BAM(128 * block.expansion,self.num_parallel)
        self.bam3 = BAM(256 * block.expansion,self.num_parallel)

        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512,  layers[3], bn_threshold, stride=2)
        self.avg = AvgParallel(nn.AvgPool2d(7, stride=1))
        self.clf_conv = ModuleParallel(nn.Linear(512 * block.expansion, num_classes))
        self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.alpha = nn.Parameter(torch.ones([1, num_parallel, 157, 157], requires_grad=True))
        self.register_parameter('alpha', self.alpha)

    def _make_layer(self, block, planes, num_blocks, bn_threshold, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel)
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l1 = self.bam1(l1)
        l2 = self.layer2(l1)
        l2 = self.bam2(l2)
        l3 = self.layer3(l2)
        l3 = self.bam3(l3)
        l4 = self.layer4(l3)
        l4 = self.dropout(l4)


        out = self.avg(l4)
        out = self.dropout(out)
        out = self.clf_conv(out)
        return out  # , alpha_soft


def refinenet(num_layers, num_classes, num_parallel, bn_threshold):
    if int(num_layers) == 18:
        layers = [2, 2, 2, 2]
    elif int(num_layers) == 34:
        layers = [3, 4, 6, 3]
    elif int(num_layers) == 50:
        layers = [3, 4, 6, 3]
    elif int(num_layers) == 101:
        layers = [3, 4, 23, 3]
    elif int(num_layers) == 152:
        layers = [3, 8, 36, 3]
    else:
        print('invalid num_layers')

    model = RefineNet(Bottleneck, layers, num_parallel, num_classes, bn_threshold)
    return model


def model_init(model, num_layers, num_parallel, imagenet=False, pretrained=True):
    if imagenet:  # 使用imagenet预训练
        key = str(num_layers) + '_imagenet'
        url = models_urls[key]
        state_dict = maybe_download(key, url)
        model_dict = expand_model_dict(model.state_dict(), state_dict, num_parallel)
        model.load_state_dict(model_dict, strict=True)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = str(num_layers) + '_' + dataset.lower()
            key = 'rf' + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def expand_model_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace('module.', '')
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = '.bn_%d' % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, '')
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict

if __name__ == '__main__':
    model = refinenet(101,1,2,(2e-2))
    print(model)