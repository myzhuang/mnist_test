#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
#from resnet20 import resnet20
from LeNet import lenet
import torch
from torch.autograd import Variable
#from torchvision.datasets import CIFAR10
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='/home2/myzhuang2/20201024_AdderNetCuda/ConvNet_mnist/dataset/')
parser.add_argument('--output_dir', type=str, default='./models_1024/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0
batch_size = 1024
#transform_train = transforms.Compose([
#    transforms.RandomCrop(32, padding=4),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])
#
#transform_test = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#])
#
#data_train = CIFAR10(args.data,
#                   transform=transform_train)
#data_test = CIFAR10(args.data,
#                  train=False,
#                  transform=transform_test)
#
#data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
#data_test_loader = DataLoader(data_test, batch_size=100, num_workers=0)
train_data = MNIST('./dataset/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.2,))]))
test_data = MNIST('./dataset/mnist',
                  train=False,
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5,), (0.2,))]))
data_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
data_test_loader = DataLoader(test_data, batch_size=256, num_workers=2)


#net = resnet20().cuda()
net = lenet().cuda()

criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    lr = 0.05 * (1+math.cos(float(epoch)/400*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
 
        output = net(images)
 
        loss = criterion(output, labels)
 
        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if i == 1:
            print('%d epoch, Loss: %f' % (epoch, loss.data.item()))
 
        loss.backward()
        optimizer.step()
 
 
def test():
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
 
    avg_loss /= len(test_data)
    acc = float(total_correct) / len(test_data)
    if acc_best < acc:
        acc_best = acc
    print('\t Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
 
 
#def train_and_test(epoch):
#    train(epoch)
#    test()
 
 
def main():
    epoch = 800
    for e in range(1, epoch+1):
        train(e)
        if e % 5 == 0:
            test()
#        train_and_test(e)
        if e % 1 == 0:
            torch.save(net,args.output_dir + 'addernet_' + str(e) + '.pkl')
 
 
if __name__ == '__main__':
    main()
