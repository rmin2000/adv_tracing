'''
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

def ResNet18Block(block, in_planes, planes, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)
    layers = []
    for s in strides:
        layers.append(block(in_planes, planes, s))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)

def ResNet18Head():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64), 
        nn.ReLU(inplace=True), 
        ResNet18Block(BasicBlock, 64, 64, 2, stride=1)
    )
    

class ResNet18Tail(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layer2 = ResNet18Block(BasicBlock, 64, 128, 2, stride=2)
        self.layer3 = ResNet18Block(BasicBlock, 128, 256, 2, stride=2)
        self.layer4 = ResNet18Block(BasicBlock, 256, 512, 2, stride=2)
        self.pool1d = nn.AdaptiveAvgPool1d(512)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.pool1d(out)
        out = self.linear(out)
        return out

