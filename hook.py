import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import pandas as pd

# 定义一个简单的卷积神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)  # 假设输入为 32x32 的图像

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x

# 创建模型实例
model = SimpleCNN()

# 注册钩子函数
def get_bn_output(module, input, output):
    # 获取通道0的输出
    channel_0_output = output[:, 0, :, :].detach().cpu().numpy()  # 将输出转换为 NumPy 数组
    print("Channel 0 Batch Norm Output Shape:", channel_0_output.shape)
    
    # 将数据写入 CSV 文件
    df = pd.DataFrame(channel_0_output.reshape(-1, channel_0_output.shape[2]))  # 展平通道
    df.to_csv('./hook_files/bn_channel_0_output.csv', index=False)

# 绑定钩子到 BN 层
hook = model.bn1.register_forward_hook(get_bn_output)

# 定义 CIFAR-10 数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 下载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 获取一个批次的数据
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 前向传播
model(images)

# 移除钩子
hook.remove()
