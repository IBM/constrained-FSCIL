#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# Taken from https://github.com/icoz69/CEC-CVPR2021/blob/main/models/resnet20_cifar.py

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import numpy as np

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.last = last

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out

class ResNet20(nn.Module):

    def __init__(self, block=BasicBlock, layers=[3,3,3], num_classes=512):


        self.inplanes = 16
        super(ResNet20, self).__init__()


        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1,
                               bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            self._make_layer(block, 16, layers[0]),
            self._make_layer(block, 32, layers[1], stride=2),
            self._make_layer(block, 64, layers[2], stride=2, last_phase=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
            
        self.n_interm_feat = 64
        self.fc = nn.Linear(self.n_interm_feat,num_classes)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, last_phase=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        if last_phase:
            for i in range(1, blocks-1):
                layers.append(block(self.inplanes, planes))
            layers.append(block(self.inplanes, planes, last=True))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_embedding(x)
        x = self.fc(x)

        return x

    def forward_conv(self,x): 
        x = self.conv_embedding(x)
        return x