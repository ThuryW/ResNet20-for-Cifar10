import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

use_square = False


def square_function(x):
    return torch.pow(x, 2)


# 定义 Fire 模块 (Squeeze + Expand)
class Fire(nn.Module):
    def __init__(self, in_ch, squeeze_ch, e1_ch, e3_ch):  # 声明 Fire 模块的超参数
        super(Fire, self).__init__()
        # Squeeze, 1x1 卷积
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1, bias=False)
        ## Expand, 1x1 卷积
        self.expand1 = nn.Conv2d(squeeze_ch, e1_ch, kernel_size=1, bias=False)
        ## Expand, 3x3 卷积
        self.expand3 = nn.Conv2d(
            squeeze_ch, e3_ch, kernel_size=3, padding=1, bias=False
        )
        if use_square:
            self.activation = square_function
        else:
            self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.activation(self.squeeze(x))
        x = torch.cat(
            [self.activation(self.expand1(x)), self.activation(self.expand3(x))], dim=1
        )
        return x


# 定义简化的 SqueezeNet 模型类
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SqueezeNet, self).__init__()
        # 50 # CONV1: 3 × 3 × 64 stride-1 convolutional layer
        # 30 # POOL
        # 80+150+200   # Fire1
        # 160+150+200   # Fire1
        # 60
        # 200+200+300            # Fire3
        # 300+200+300            # Fire4
        # 80 # CONV2
        # 10
        # = 2670
        # slot: 8192/(32/3)/(32/3) = 67
        # TP: 67 / 2670 = 25.1

        # https://github.com/dacorvo/tensorsandbox/tree/master/cifar10/models/squeeze
        # conv [3x3x3x64]
        # pool x2
        # fire 32/64/64
        # fire 32/64/64
        # pool x2
        # fire 32/128/128
        # fire 32/128/128
        # conv [1,1,128*2,10]
        # average pool
        # Size : 0.15 Millions of parameters
        # Flops: 22.84 Millions of operations
        self.C1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.F1 = Fire(64, 16, 64, 64)
        self.P1 = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.F2 = Fire(128, 32, 64, 64)
        self.P2 = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.F3 = Fire(128, 64, 128, 128)
        self.P3 = nn.AvgPool2d(kernel_size=3, stride=2, ceil_mode=True)

        _out_F4 = 256
        self.F4 = Fire(256, 64, _out_F4, _out_F4)
        self.D4 = nn.Dropout(p=0.2)

        # self.C5 = nn.Conv2d(_out_F4 * 2, num_classes, kernel_size=1, bias=False)
        self.C4 = nn.Conv2d(_out_F4 * 2, num_classes, kernel_size=1, bias=False)
        if use_square:
            self.R5 = square_function
        else:
            self.R5 = nn.ReLU(inplace=True)
        self.A5 = nn.AdaptiveAvgPool2d((1, 1))
        # '''

        # self.debug = True
        self.debug = False

    def forward(self, x):
        print_shape(self.debug, x, "input")
        x = self.C1(x)
        print_shape(self.debug, x, "C1")
        x = self.F1(x)
        print_shape(self.debug, x, "F1")
        x = self.P1(x)
        print_shape(self.debug, x, "P1")

        x = self.F2(x)
        print_shape(self.debug, x, "F2")
        x = self.P2(x)
        print_shape(self.debug, x, "P2")

        x = self.F3(x)
        print_shape(self.debug, x, "F3")
        x = self.P3(x)
        print_shape(self.debug, x, "P3")

        x = self.F4(x)
        print_shape(self.debug, x, "F4")
        x = self.D4(x)
        print_shape(self.debug, x, "D4")

        # x = self.C5(x)
        x = self.C4(x)
        print_shape(self.debug, x, "C5")
        x = self.R5(x)
        print_shape(self.debug, x, "R5")
        x = self.A5(x)
        print_shape(self.debug, x, "A5")

        out = x.view(x.size(0), -1)
        print_shape(self.debug, out, "out")
        self.debug = False
        return out


def print_shape(debug, x, in_str=""):
    if debug:
        print(f"{in_str}: {x.shape}")
        # print(f"{x}")

'''
my_datasets = datasets.CIFAR10
epochs = 1000
batch_size = 256
lr = 0.00001
weight_decay = 5e-4
amsgrad = True
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.Resize(32),  # 图像大小调整为 (w,h)=(32，32)
        transforms.ToTensor(),  # 将图像转换为张量 Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ]
)
model = SqueezeNet(num_classes=10)
'''

if __name__ == "__main__":
    from thop import profile

    sample = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(sample,))
    print(f"FLOPS: {flops}; PARAMS: {params}")
