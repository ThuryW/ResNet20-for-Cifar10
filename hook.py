import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from model.resnet import *

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = resnet32()
print(f"Using device: {device}")

# 加载训练后的模型权重
checkpoint = torch.load('/home/wangtianyu/pytorch_resnet_cifar10/save_resnet32/model.th', map_location=device)

model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model.load_state_dict(checkpoint['net']) 
else:
    # 如果是 DataParallel 模型，移除 'module.' 前缀
    # state_dict = checkpoint['net']
    state_dict = checkpoint['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict) 

model.eval()  # 切换到评估模式

# 注册钩子函数
def get_linear_input(module, input, output):
    # 打印输入的形状和通道数
    input_tensor = input[0]  # 获取输入张量
    input_shape = input_tensor.shape
    num_features = input_shape[1]  # 对于线性层，特征数是第二个维度
    print("Input Shape:", input_shape)
    print("Number of Features:", num_features)

    # 获取每个特征的输入值
    for feature in range(num_features):
        feature_input = input_tensor[:, feature].detach().cpu().numpy()  # 获取每个特征列的输入
        # print(f"Feature {feature} Input Shape:", feature_input.shape)

        # 计算输入的均值、方差、最大值和最小值
        mean = feature_input.mean()
        variance = feature_input.var()
        max_value = feature_input.max()
        min_value = feature_input.min()
        
        print(f"Feature {feature} Input Max: {max_value:.4f}, Min: {min_value:.4f}, Mean: {mean:.4f}, Variance: {variance:.4f}")

def get_layer_output(module, input, output):
    # 打印输出的形状和通道数
    output_shape = output.shape
    num_channels = output_shape[1]  # 通道数是第二个维度
    print("Layer Output Shape:", output_shape)
    print("Number of Channels:", num_channels)

    # 获取通道0的输出
    for channel in range(num_channels):
        channel_output = output[:, channel, :, :].detach().cpu().numpy()  # 将输出转换为 NumPy 数组

        # 计算输出的均值、方差、最大值和最小值
        mean = channel_output.mean()
        variance = channel_output.var()
        max_value = channel_output.max()
        min_value = channel_output.min()
        
        print(f"Channel {channel} Output Max: {max_value:.4f}, Min: {min_value:.4f}, Mean: {mean:.4f}, Variance: {variance:.4f}")

def get_layer_output_all(module, input, output):
    # 获取输出的形状
    output_tensor = output
    output_shape = output_tensor.shape
    print("Layer Output Shape:", output_shape)

    # 将输出张量展平为一维
    flattened_output = output_tensor.view(-1).detach().cpu().numpy()  # 展平张量并转为 NumPy 数组

    # 计算整个输出的最大值、最小值、均值和方差
    max_value = flattened_output.max()
    min_value = flattened_output.min()
    mean = flattened_output.mean()
    variance = flattened_output.var()

    print(f"Output Max: {max_value:.4f}, Min: {min_value:.4f}, Mean: {mean:.4f}, Variance: {variance:.4f}")


def get_layer_input(module, input, output):
    # 打印输入的形状和通道数
    input_shape = input[0].shape  # input 是一个元组，第一个元素是输入张量
    num_channels = input_shape[1]  # 通道数是第二个维度
    print("Layer Input Shape:", input_shape)
    print("Number of Channels:", num_channels)

    # 获取通道0的输入
    for channel in range(num_channels):
        channel_input = input[0][:, channel, :, :].detach().cpu().numpy()  # 将输入转换为 NumPy 数组

        # 计算输入的均值、方差、最大值和最小值
        mean = channel_input.mean()
        variance = channel_input.var()
        max_value = channel_input.max()
        min_value = channel_input.min()
        
        print(f"Channel {channel} Input Max: {max_value:.4f}, Min: {min_value:.4f}, Mean: {mean:.4f}, Variance: {variance:.4f}")

def get_layer_input_all(module, input, output):
    # 获取输入的形状
    input_tensor = input[0]
    input_shape = input_tensor.shape
    print("Layer Input Shape:", input_shape)

    # 将输入张量展平为一维
    flattened_input = input_tensor.view(-1).detach().cpu().numpy()  # 展平张量并转为 NumPy 数组

    # 计算整个输入的最大值、最小值、均值和方差
    max_value = flattened_input.max()
    min_value = flattened_input.min()
    mean = flattened_input.mean()
    variance = flattened_input.var()

    print(f"Input Max: {max_value:.4f}, Min: {min_value:.4f}, Mean: {mean:.4f}, Variance: {variance:.4f}")


# 绑定钩子到 BN 层
if device == 'cuda':
    hook = model.module.layer1[1].bn1.register_forward_hook(get_layer_output)
else:
    # hook = model.layer3[2].bn2.register_forward_hook(get_layer_output_all)
    hook = model.layer3[2].conv1.register_forward_hook(get_layer_input_all)


# 定义 CIFAR-10 数据集的转换
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 下载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

# # 选择前 227 张图片
# start_image = 2497
# subset_indices = list(range(start_image, start_image + 227))
# train_subset = Subset(train_dataset, subset_indices)
# train_loader = DataLoader(train_subset, batch_size=227, shuffle=False)

# 使用整个数据集进行推理
train_loader = DataLoader(train_dataset, batch_size=10000, shuffle=False)

# 获取一个批次的数据
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 将数据移动到指定设备
images = images.to(device)

# 前向传播
with torch.no_grad():  # 不计算梯度
    model(images)

# 移除钩子
hook.remove()
