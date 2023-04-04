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

class ChannelAttention(nn.Module):
    def __init__(self,inplanes,radio=16):
        super(ChannelAttention,self).__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.fc1=nn.Conv2d(inplanes,inplanes//radio,1,bias=False)
        self.relu1=nn.ReLU()
        self.fc2=nn.Conv2d(inplanes//radio,inplanes,1,bias=False)

        self.sigmoid=nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avg_out=self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out=self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out=avg_out+max_out
        return self.sigmoid(out)

class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""
    def __init__(self, in_dim, activation=None, size=56, ratio=16, ret_att=False):
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # bias默认为True
        self.query_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.query_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        # self.gamma1 = nn.Parameter(torch.ones(1))

        # self.query_conv2 = nn.Conv2d(
        #     in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        # self.key_conv2 = nn.Conv2d(
        #     in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        # self.value_conv2 = nn.Conv2d(
        #     in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma2 = nn.Parameter(torch.zeros(1))

        self.linear1 = nn.Linear(size * size, size * size)
        # self.linear2 = nn.Linear(size * size, size * size)
        self.convblk = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU()
        )

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def channel_shuffle(self, x, groups):  # group=2
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)  # (b, -1, groups, h, w)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        B, C, H, W = x.size()

        def _get_att(a, b):
            aggr = torch.cat((a, b), dim=1)
            # channel shuffle unit
            # aggr = self.convblk(aggr)
            aggr = self.convblk(self.channel_shuffle(aggr, groups=2))
            # q,k,v，y on x
            proj_query1 = self.query_conv1(a).view(
                B, -1, H * W).permute(0, 2, 1)  # B, HW, C
            proj_key1 = self.key_conv1(aggr).view(
                B, -1, H * W)  # B X C x (*W*H)
            energy1 = torch.bmm(proj_query1, proj_key1)  # B, HW, HW
            attention1 = self.softmax(self.linear1(energy1))
            # q,k,v，x on y
            proj_query2 = self.query_conv2(b).view(
                B, -1, H * W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv1(aggr).view(
                B, -1, H * W)  # B X C x (*W*H)
            energy2 = torch.bmm(proj_query2, proj_key2)  # B, HW, HW
            attention2 = self.softmax(self.linear1(
                energy2))  # BX (N) X (N)
            return attention1, attention2,aggr

        att_y_on_x, att_x_on_y,aggr = _get_att(x, y)
        proj_value_y_on_x = self.value_conv1(aggr).view(
            B, -1, H * W)  # B, C, HW
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)  # R
        out_x = self.gamma1 * out_y_on_x + x  # T'

        proj_value_x_on_y = self.value_conv1(aggr).view(
            B, -1, H * W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)  # Rh
        out_y = self.gamma1 * out_x_on_y + y  # Th'

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention

class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=512, out_chan=512,num_classes=1,expansion=4, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan*2, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, radio=16)
        self.clf = nn.Linear(512 * expansion, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)

        out = F.adaptive_avg_pool2d(fuse_fea, (1, 1))
        out = out.view(x.size(0), -1)
        out = self.dropout(out)
        out = self.clf(out)

        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


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
        self.dual_cma1 = DualCrossModalAttention(in_dim=64*block.expansion,size=56, ret_att=False)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.dual_cma2 = DualCrossModalAttention(in_dim=128*block.expansion,size=28, ret_att=False)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.dual_cma3 = DualCrossModalAttention(in_dim=256*block.expansion,size=14, ret_att=False)
        self.layer4 = self._make_layer(block, 512,  layers[3], bn_threshold, stride=2)
        # self.dual_cma4 = DualCrossModalAttention(in_dim=512*block.expansion,size=7, ret_att=False)
        # self.fusion = FeatureFusionModule(in_chan=512*block.expansion,out_chan=512*block.expansion,num_classes=num_classes,expansion=block.expansion)
        self.avg = AvgParallel(nn.AvgPool2d(7, stride=1))
        self.clf_conv = ModuleParallel(nn.Linear(512 * block.expansion, num_classes))
        # self.alpha = nn.Parameter(torch.ones(num_parallel, requires_grad=True))
        # self.alpha = nn.Parameter(torch.ones([1, num_parallel, 157, 157], requires_grad=True))
        # self.register_parameter('alpha', self.alpha)

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
        da1 = self.dual_cma1(l1[0],l1[1])
        l2 = self.layer2(da1)
        da2 = self.dual_cma2(l2[0], l2[1])
        l3 = self.layer3(da2)
        da3 = self.dual_cma3(l3[0], l3[1])
        l4 = self.layer4(da3)
        # da4 = self.dual_cma4(l4[0], l4[1])
        l4 = self.dropout(l4)

        out = self.avg(l4)
        out = self.dropout(out)
        out = self.clf_conv(out)
        # out = self.fusion(l4[0],l4[1])
        return out # , alpha_soft


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
    model = refinenet(50,1,2,(2e-2))
    dummy = torch.rand((1, 3, 224, 224))
    inputs = [dummy, dummy]
    out = model(inputs)
    print(out)
    # print(model)