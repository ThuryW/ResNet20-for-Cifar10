'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from model.resnet20 import *
from utils import progress_bar, kaiming_initialization

parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Training')
parser.add_argument('--opt', default = 'adam', type = str, help = 'sgd or adam or adamw')
parser.add_argument('--scheduler', default = 'no', type = str, help = 'no or cos or step')
parser.add_argument('--lr', default = 0.001, type = float, help = 'learning rate')
parser.add_argument('--batch_size', default = 256, type = int, help = 'train batch size')
parser.add_argument('--ep', default = 200, type = int, help = 'epoch')
parser.add_argument('--wd', default = 5e-4, type = float, help = 'weight decay')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
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
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


trainset = torchvision.datasets.CIFAR10(root = './data', train = True, transform = transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers = 2)

testset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size = 512, shuffle = False, num_workers = 2)

# Model
print('==> Building model..')

net = resnet20()
kaiming_initialization(net)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

if args.opt == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
elif args.opt == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.opt == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
else:
    raise ValueError("Wrong optimizer!")

if args.scheduler == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ep)
elif args.scheduler == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)
else:
    scheduler = None

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

for epoch in range(start_epoch, start_epoch + args.ep):
    # Training
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

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    # Test
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print(f'Save best model @ Epoch {epoch}')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc 