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
from models.cen_modules import *
# from utils.shape_conv import ShapeConv2d


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
    # return ConvParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
    #                                 stride=stride, padding=1, bias=bias),
    #                     ShapeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
    #                  padding=1, bias=False))

def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))
    # return ConvParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
    #                                 stride=stride, padding=0, bias=bias),
    #                     ShapeConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
    #                  padding=0, bias=False))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
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

        self.exchange = Exchange()
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
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out

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
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
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

    # model = RefineNet(BasicBlock, layers, num_parallel, num_classes, bn_threshold)
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