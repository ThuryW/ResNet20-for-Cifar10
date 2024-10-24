import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from model.resnet20 import *
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = resnet20()

# 加载训练后的模型权重
checkpoint = torch.load('./checkpoint/ckpt.pth')

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

model.load_state_dict(checkpoint['net']) 
model.eval()  # 切换到评估模式

# 注册钩子函数
def get_bn_output(module, input, output):
    # 获取通道0的输出
    channel_0_output = output[:, 3, :, :].detach().cpu().numpy()  # 将输出转换为 NumPy 数组
    print("Channel 0 Batch Norm Output Shape:", channel_0_output.shape)

    # 计算输出的均值和方差
    mean = channel_0_output.mean()
    variance = channel_0_output.var()
    print(f"Channel 0 Batch Norm Output Mean: {mean:.4f}, Variance: {variance:.4f}")
    
    # 将数据写入 CSV 文件
    df = pd.DataFrame(channel_0_output.reshape(-1, channel_0_output.shape[2]))  # 展平通道
    df.to_csv('./hook_files/bn_channel_0_output.csv', index=False)

# 绑定钩子到 BN 层
if device == 'cuda':
    hook = model.module.bn1.register_forward_hook(get_bn_output)
else:
    hook = model.bn1.register_forward_hook(get_bn_output)


# 定义 CIFAR-10 数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
])

# 下载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 获取一个批次的数据
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 前向传播
with torch.no_grad():  # 不计算梯度
    model(images)

# 移除钩子
hook.remove()
