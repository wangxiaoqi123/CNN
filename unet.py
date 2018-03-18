

       from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
import torch
from numpy.linalg import svd
from numpy.random import normal
from math import sqrt


class UNet(nn.Module):

    def __init__(self, colordim=1):
        super(UNet, self).__init__()
        # input of (n,n,1), output of (n-2,n-2,64)
        self.conv1 = nn.Conv2d(colordim, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn2_out = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn3_out = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn4_out = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn5_out = nn.BatchNorm2d(1024)
        self.conv6 = nn.Conv2d(1024, colordim, 1, 1, 0)
        self.bn6 = nn.BatchNorm2d(colordim)
        self._initialize_weights()

    def forward(self, x1):
        x1 = F.relu(self.conv1(x1))

        x2 = F.relu(self.bn2(self.conv2(x1)))

        x3 = self.bn2_out(torch.cat((x1, x2), 1))  

        x4 = F.relu(self.bn3(self.conv3(x3)))

        x5 = self.bn3_out(torch.cat((x3, x4), 1)) 

        x6 = F.relu(self.bn4(self.conv4(x5)))

        x7=self.bn4_out(torch.cat((x5, x6), 1)) 

        x8 = F.relu(self.bn5(self.conv5(x7)))

        x9=self.bn5_out(torch.cat((x7, x8), 1))  

        x10= F.relu(self.conv6(x9))

        return F.softsign(self.bn6(x10))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n=m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

unet=UNet()
# unet = UNet().cuda()
print(unet)
