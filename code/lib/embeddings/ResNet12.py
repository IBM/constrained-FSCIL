#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# This is from https://github.com/icoz69/DeepEMD/blob/master/Models/models/resnet.py, 10/09/2021
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0


    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class ResNet12(nn.Module):

    def __init__(self, args): 
        super(ResNet12, self).__init__()

        # variables copied from function parameters
        block = BasicBlock
        self.in_planes = args.image_size[0]
        out_planes = list(args.num_filters) # default: [64, 160, 320, 640]
        dim_features = args.dim_features
        dropout_final = args.dropout_rate
        dropout_interm = args.dropout_rate_interm

        self.conv_embedding = nn.Sequential(
            self._make_layer(block, out_planes[0], stride=2, drop_rate=dropout_interm),
            self._make_layer(block, out_planes[1], stride=2, drop_rate=dropout_interm),
            self._make_layer(block, out_planes[2], stride=2, drop_rate=dropout_interm),
            self._make_layer(block, out_planes[3], stride=2, drop_rate=dropout_interm),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_final, inplace=False))

        self.n_interm_feat = out_planes[-1]
        self.fc = nn.Linear(self.n_interm_feat,dim_features)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, drop_rate))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_embedding(x)
        x = self.fc(x)
        return x

    def forward_conv(self,x): 
        x = self.conv_embedding(x)
        return x