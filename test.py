import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from model.resnet import *
from utils import AvgMeter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_test_data(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(root = './data', train = False, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, shuffle = False, num_workers = 2)

    return testloader

def test_process(args, test_loader):
    net = resnet32()
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    checkpoint = torch.load(args.path)
    net.load_state_dict(checkpoint['net'])
    best_epoch = checkpoint['epoch']
    print('best epoch:', best_epoch)

    net.eval()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    for image_batch, gt_batch in tqdm(test_loader):
        image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
        gt_batch = gt_batch.long()
        with torch.no_grad():
            pred_batch = net(image_batch)
            loss = criterion(pred_batch, gt_batch)
        loss_meter.add(loss.item(), image_batch.size(0))
        acc = (pred_batch.argmax(dim=-1).long() == gt_batch).float().mean()
        acc_meter.add(acc.item(), image_batch.size(0))
    test_loss = loss_meter.avg()
    test_acc = acc_meter.avg()

    return test_loss, test_acc

def test_result(start_time, test_loss, test_acc):
    l_str = 'Elapsed time {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
    print(l_str)
    l_str = '***** Test set result ***** loss: {:1.4f}, accuracy {:1.2f}%'.format(test_loss, test_acc * 100)
    print(l_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Test')
    parser.add_argument('--batch_size', default = 256, type = int, help = 'test batch size')
    parser.add_argument('--path', default = '/home/wangtianyu/my_resnet20/checkpoint/ckpt.pth', type = str, help = 'test model path')
    args = parser.parse_args()

    test_loader = load_test_data(args)

    start_time = time.time()
    test_loss, test_acc = test_process(args, test_loader)
    
    test_result(start_time, test_loss, test_acc)
