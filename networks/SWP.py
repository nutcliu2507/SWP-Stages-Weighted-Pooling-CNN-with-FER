from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
import math


class SWP(nn.Module):
    def __init__(self, num_class=7, pretrained=False):
        super(SWP, self).__init__()

        # here pretrained is image-Net weight to backbone
        resnet = models.resnet18(pretrained)

        # here pretrained is MS-celeb-1M weight to backbone
        if pretrained:
            checkpoint = torch.load('./models/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)

        # Divided of Resnet with stages
        self.Res_s2 = nn.Sequential(*list(resnet.children())[:-4])
        self.Res_s3 = nn.Sequential(*list(resnet.children())[-4])
        self.Res_s4 = nn.Sequential(*list(resnet.children())[-3])

        # spatial attention
        self.sa_x2 = SpatialAttention2()
        self.sa_x3 = SpatialAttention3()
        self.sa_x4 = SpatialAttention()

        # flatten
        self.gap = nn.AdaptiveAvgPool2d(1)

        # normalization
        self.bn = nn.BatchNorm1d(num_class)

        # SW-MLP
        self.MLP_s2 = MLP_s2()
        self.MLP_s3 = MLP_s3()
        self.MLP_s4 = MLP_s4()

        # softmax
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        # stages Features
        x = self.Res_s2(x)
        x2= x
        x = self.Res_s3(x)
        x3= x
        x4 = self.Res_s4(x)

        # attention
        x2= self.sa_x2(x2)
        x3= self.sa_x3(x3)
        x4= self.sa_x4(x4)

        #flatten
        x2 = self.gap(x2)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.gap(x3)
        x3 = x3.view(x3.size(0), -1)
        x4 = self.gap(x4)
        x4 = x4.view(x4.size(0), -1)

        # out
        out_x2 = self.MLP_s2(x2)
        out_x3 = self.MLP_s3(x3)
        out_x4 = self.MLP_s4(x4)
        out_x4 = self.bn(out_x4)

        out1 = out_x4
        # add
        out_x4 = torch.add(1*out_x4, 0.3*out_x2, alpha=10)
        out_x4 = torch.add(out_x4, 0.5*out_x3, alpha=10)
        # softmax
        # out_x4 = self.softmax(out_x4)

        return out1, out_x4

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out
class SpatialAttention2(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(128),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out
class SpatialAttention3(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(256),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1, keepdim=True)
        out = x * y

        return out

class MLP_s2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            GELU(),
            nn.Linear(128, 64),
            GELU(),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.layers(x)
class MLP_s3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 128),
            GELU(),
            nn.Linear(128, 64),
            GELU(),
            nn.Linear(64, 1))
    def forward(self, x):
        return self.layers(x)
class MLP_s4(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            GELU(),
            nn.Linear(512, 256),
            GELU(),
            nn.Linear(256, 128),
            GELU(),
            nn.Linear(128, 64),
            GELU(),
            nn.Linear(64, 7),)
    def forward(self, x):
        return self.layers(x)


