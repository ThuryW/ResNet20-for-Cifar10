import os
import sys
import time
import logging

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.nn as nn
import torch.nn.init as init
from torch.utils.tensorboard import SummaryWriter

def data_loader(dataset_dir='./data', train_ratio=0.8, batch_size=256, enhance=False):
    """
    :param dataset_dir: the directory to restore dataset
    :param train_ratio: the ratio of training set in train_val_set
    :param batch_size: the batch size during training
    :return: dataset for training, validation and testing
    If the dataset hasn't been prepared, it will be downloaded according to download=True.
    """

    print('==> Loading data...')
    if enhance == False:
        transform_train = transforms.Compose([  
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([  
            transforms.RandomCrop(32, padding = 4), 
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_val_set = torchvision.datasets.CIFAR10(root = dataset_dir, train = True, transform = transform_train)
    test_set = torchvision.datasets.CIFAR10(root = dataset_dir, train = False, transform = transform_test)

    # split train_set and val_set
    total_len = len(train_val_set)
    train_len = int(total_len * train_ratio)
    val_len = total_len - train_len
    train_set, val_set = data.random_split(train_val_set, lengths=[train_len, val_len])
    print(f'Size of Training set: {len(train_set)}')
    print(f'Size of Validation set: {len(val_set)}')
    print(f'Size of Testing set: {len(test_set)}')

    # generate loaders
    # use shuffle=True to introduce randomness during training
    train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    val_loader = data.DataLoader(val_set, batch_size = batch_size, shuffle = False)
    test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

    return train_loader, val_loader, test_loader


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def print_and_logger(string, logger):
    print(string)
    logger.info(string)


def init_logger(args):
    log_dir = './log/log_{}'.format(time.strftime("%Y%m%d%H%M%S", time.localtime()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    f = logging.FileHandler(os.path.join(log_dir, 'log.txt'), mode='w+')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    f.setFormatter(formatter)
    logger.addHandler(f)

    l_str = '> Arguments:'
    print_and_logger(l_str, logger)
    for key in args.__dict__.keys():
        l_str = '\t{}: {}'.format(key, args.__dict__[key])
        print_and_logger(l_str, logger)

    return logger, log_dir


class DictTable(dict):
    # Overridden dict class which takes a dict in the form {'a': 2, 'b': 3},
    # and renders an HTML Table in IPython Notebook.
    def _repr_html_(self):
        html = ["<table width=100%>"]
        for key, value in self.items():
            html.append("<tr>")
            html.append("<td>{0}</td>".format(key))
            html.append("<td>{0}</td>".format(value))
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)


def init_logger_writer(args):
    logger, log_dir = init_logger(args)
    writer = SummaryWriter(log_dir)
    writer.add_text('args', DictTable(args.__dict__)._repr_html_(), 0)  # record the training configurations

    return logger, writer


def inform_logger_writer(logger, writer, epoch, train_loss, val_loss, train_acc, val_acc, time_list):
    l_str = '[Epoch {}] elapsed time: {:1.4f}s, train time: {:1.4f}s, val time: {:1.4f}s'. \
        format(epoch, time_list[2] - time_list[0], time_list[1] - time_list[0], time_list[2] - time_list[1])
    print_and_logger(l_str, logger)
    l_str = '***** Training set result ***** loss: {:1.4f}, accuracy {:1.2f}%'.format(train_loss, train_acc * 100)
    print_and_logger(l_str, logger)
    l_str = '***** Validation set result ***** loss: {:1.4f}, accuracy {:1.2f}%'.format(val_loss, val_acc * 100)
    print_and_logger(l_str, logger)

    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('train/accuracy', train_acc * 100, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('val/accuracy', val_acc * 100, epoch)


def close_logger_writer(logger, writer, start_time, test_loss, test_acc):
    print_and_logger(' ', logger)
    l_str = 'Elapsed time {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
    print_and_logger(l_str, logger)
    l_str = '***** Test set result ***** loss: {:1.4f}, accuracy {:1.2f}%'.format(test_loss, test_acc * 100)
    print_and_logger(l_str, logger)

    loggers = list(logger.handlers)
    for i in loggers:
        logger.removeHandler(i)
        i.flush()
        i.close()
    writer.close()

class AvgMeter(object):
    def __init__(self):
        self.cnt = 0
        self.sum = 0

    def add(self, x, cnt=1):
        self.sum += x * cnt
        self.cnt += cnt

    def avg(self):
        assert self.cnt != 0
        return self.sum / self.cnt
    
def kaiming_initialization(model):
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # Kaiming 初始化对于卷积层
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # 初始化偏置为0
        elif isinstance(layer, nn.Linear):
            # Kaiming 初始化对于全连接层
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # 初始化偏置为0