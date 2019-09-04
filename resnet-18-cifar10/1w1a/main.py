'''Train CIFAR10 with PyTorch.'''
import torch
import math
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--Tmin', default=0.1, type=float)
parser.add_argument('--Tmax', default=10, type=float)

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min = 0, last_epoch=-1)
#lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350], last_epoch=start_epoch - 1)
print (net.module)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        print (batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Lr: %.2f'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, optimizer.param_groups[0]['lr']))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print (batch_idx, len(testloader), 'Loss: %.3f | TestAcc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):

    def Log_UP(K_min, K_max, E):
        Kmin, Kmax = math.log(K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * E)]).float().cuda()
    T_min, T_max = args.Tmin, args.Tmax
    t = Log_UP(T_min, T_max, epoch)
    if (t < 1):
        k = 1 / t
    else:
        k = torch.tensor([1]).float().cuda()
    #print (net.module)
    for i in range(2):

        # net.module.layers[i].conv1.k = k
        # net.module.layers[i].conv2.k = k
        # net.module.layers[i].conv3.k = k
        # net.module.layers[i].conv1.t = t
        # net.module.layers[i].conv2.t = t
        # net.module.layers[i].conv3.t = t

        net.module.layer1[i].conv1.k = k
        net.module.layer1[i].conv2.k = k
        net.module.layer1[i].conv1.t = t
        net.module.layer1[i].conv2.t = t

        net.module.layer2[i].conv1.k = k
        net.module.layer2[i].conv2.k = k
        net.module.layer2[i].conv1.t = t
        net.module.layer2[i].conv2.t = t

        net.module.layer3[i].conv1.k = k
        net.module.layer3[i].conv2.k = k
        net.module.layer3[i].conv1.t = t
        net.module.layer3[i].conv2.t = t

        net.module.layer4[i].conv1.k = k
        net.module.layer4[i].conv2.k = k
        net.module.layer4[i].conv1.t = t
        net.module.layer4[i].conv2.t = t

    train(epoch)
    lr_scheduler.step()
    test(epoch)


