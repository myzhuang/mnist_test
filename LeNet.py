# 2020.01.10-Replaced conv with adder
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

#import adder
import torch.nn as nn
import torch.nn.functional as F
import torch


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv5x5(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)
    
class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        self.conv1 = conv3x3(1, 6, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = conv3x3(6, 16, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(1024, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.max_pool2d(h, (2,2))
        h = self.conv2(h)
        h = self.bn2(h)
        h = F.max_pool2d(h, (2,2))
        h = h.reshape(-1, 1024)
        h = self.fc1(h)#1024->120
        h = F.relu(h)
        h = self.fc2(h)#120->84
        h = F.relu(h)
        h = self.fc3(h)#84->10
        return h
        
#    def forward(self, x):
#        x = F.max_pool2d(self.bn1(self.conv1(x)), (2,2))
#        x = F.max_pool2d(self.bn2(self.conv2(x)), (2,2))
#        print('aa', x.shape)
#        x = x.reshape(-1, self.num_flat_features(x))
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = x.reshape(-1, self.num_flat_features(x))
#        x = self.fc3(x)
#        return x
#    def num_flat_features(self, x):
#        size = x.size()[1:]
#        num_features = 1
#        for s in size:
#            num_features *= s
#        return num_features


if __name__ == '__main__':
    net = lenet()
    x = torch.randn(1, 1, 32, 32, requires_grad=True)
    a = net(x)

   