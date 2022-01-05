#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:39:50 2021

@author: shan
"""

import torchvision.models as models
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self, num_classes=16, win_size=32, fix_para_layer=10) :
        super(VGG16, self).__init__()        
        vgg16 = models.vgg16(pretrained=True)
        
        if win_size == 32:
            conv1 = nn.Conv2d(512, 128, 3, padding=1)
            conv2 = nn.Conv2d(128, num_classes, 3, padding=1)
            self.model = nn.Sequential(vgg16.features[:24], conv1, nn.ReLU(), conv2) #[B,num_classes,32,32]
        elif win_size == 16:
            conv1 = nn.Conv2d(512, 128, 3, padding=1)
            conv2 = nn.Conv2d(128, num_classes, 3, padding=1)
            self.model = nn.Sequential(vgg16.features, conv1, nn.ReLU(), conv2) #[B,num_classes,16,16]
        else:
        # elif win_size == 8:
            conv1 = nn.Conv2d(512, 128, 3, stride=2, padding=1)
            conv2 = nn.Conv2d(128, num_classes, 3, padding=1)
            self.model = nn.Sequential(vgg16.features, conv1, nn.ReLU(), conv2) #[B,num_classes,8,8]
        for i in range(fix_para_layer):
            for param in self.model[0][i].parameters():
                param.requires_grad = False
    def forward(self, x):
        return self.model(x)


