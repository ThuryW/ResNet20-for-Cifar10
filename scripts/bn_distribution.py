import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
import argparse # 导入 argparse 模块

# 确保你可以从 resnet.py 导入 resnet20 模型
# 如果 resnet.py 不在同一目录下，或者 resnet20 的导入方式不同，请相应调整
from resnet import resnet20 # 假设你的 ResNet20 模型定义在这里

# 导入 CIFAR-10 数据集相关的库
import torchvision
import torchvision.transforms as transforms

def visualize_bn_distributions(args):
    """
    可视化 ResNet20 模型中所有 BN 层在给定 batch_size 输入下的数据分布。

    Args:
        args (argparse.Namespace): 包含所有配置参数的对象。
    """
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    model = resnet20()
    try:
        checkpoint = torch.load(args.pth_file_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Successfully loaded model weights: {args.pth_file_path}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        print("Please ensure your .pth file matches the ResNet20 model structure.")
        return None

    model.eval() # 设置为评估模式，这将关闭 BN 层的训练模式（即使用均值和方差）

    # 用于保存 BN 层的输出
    bn_outputs = OrderedDict()

    # 注册 forward hook 来捕获 BN 层的输出
    def hook_fn(module, input, output):
        # output 是一个 Tensor，将其从 GPU 转移到 CPU 并转换为 NumPy 数组
        bn_outputs[module.name].append(output.detach().cpu())

    # 遍历模型的所有模块，找到 BatchNorm2d 层并注册 hook
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.name = name # 给模块一个名字，方便识别
            bn_outputs[name] = [] # 初始化列表来存储该 BN 层的输出
            module.register_forward_hook(hook_fn)

    # --- 使用 CIFAR-10 数据集 ---
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # CIFAR-10 标准化参数
    ])

    # 加载 CIFAR-10 测试数据集
    # download=True 会在首次运行时下载数据集到 ./data 目录
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"开始使用 CIFAR-10 数据集进行前向传播，总批次数量：{len(testloader)}")

    # 遍历数据集并执行前向传播以触发 hook
    with torch.no_grad(): # 在评估模式下，不需要计算梯度
        for i, (inputs, labels) in enumerate(testloader):
            _ = model(inputs)
            if (i + 1) % 100 == 0: # 每处理100个批次打印一次进度
                print(f"  Processed {i + 1}/{len(testloader)} batches.")
            # 考虑只处理前N个batch，以避免处理整个数据集导致内存过大或时间过长
            # 例如： if i >= 10: break # 只处理前10个batch

    print(f"前向传播完成。共捕获到 {len(bn_outputs)} 个 BN 层的输出。")

    # 绘制并保存每个 BN 层的输出分布
    for bn_layer_name, outputs_list in bn_outputs.items():
        if not outputs_list:
            print(f"警告：BN 层 '{bn_layer_name}' 没有捕获到输出，可能没有被前向传播激活。")
            continue

        # 将所有 batch 的输出连接起来
        # 形状: (total_samples, num_channels, height, width)
        all_outputs_tensor = torch.cat(outputs_list, dim=0) 

        # 展平 HxW 维度，以便按通道查看分布
        # 形状: (num_channels, total_samples * height * width)
        data_to_plot = all_outputs_tensor.permute(1, 0, 2, 3).reshape(all_outputs_tensor.shape[1], -1).numpy()
        
        num_channels = data_to_plot.shape[0]

        # 计算合适的子图布局
        cols = int(np.ceil(np.sqrt(num_channels)))
        rows = int(np.ceil(num_channels / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        # 展平 axes 数组，方便迭代
        axes = axes.flatten() if num_channels > 1 else np.array([axes])

        for i in range(num_channels):
            ax = axes[i]
            # 绘制直方图
            ax.hist(data_to_plot[i], bins=50, density=True, alpha=0.7, color='skyblue')
            ax.set_title(f'Channel {i+1}')
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Distribution of BN Layer: {bn_layer_name}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局以避免标题重叠
        
        # 保存图片
        filename = os.path.join(args.save_dir, f'{bn_layer_name.replace(".", "_")}_distribution.jpg')
        plt.savefig(filename, dpi=150)
        plt.close(fig) # 关闭图形，释放内存
        print(f"已保存 {bn_layer_name} 的分布图到 {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize Batch Normalization layer distributions in ResNet20.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/base_models/20_ckpt_92.23.pth",
                        help='Path to the .pth model file.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Input batch size for data loading.')
    parser.add_argument('--save_dir', type=str, 
                        default="/home/wangtianyu/relu_finetune/hook/bn_distribution",
                        help='Directory to save the distribution images.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')

    args = parser.parse_args()

    # 调用可视化函数
    visualize_bn_distributions(args)

    print("\n所有 BN 层的分布图已生成并保存到指定目录。")