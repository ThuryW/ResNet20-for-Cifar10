import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
from resnet import resnet20 # 确保可以从 resnet.py 导入 resnet20 模型
import math

def get_relu_input_distributions_per_channel(pth_file_path, batch_size, data_dir, save_dir):
    """
    获取并可视化所有 ReLU 层的输入数据分布，每层一张大图，按通道划分多个子图。
    """
    # 1. 加载模型
    model = resnet20()
    try:
        checkpoint = torch.load(pth_file_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✨ 成功加载模型权重：{pth_file_path}")
    except Exception as e:
        print(f"❌ 加载模型权重失败：{e}")
        print("请确保你的 .pth 文件与 ResNet20 模型结构匹配。")
        return

    model.eval()

    # 2. 注册 Hook 来捕获所有 ReLU 层的输入
    relu_input_data = OrderedDict()
    
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_input_data[name] = []
            
            def hook_fn(module, input, name=name):
                relu_input_data[name].append(input[0].detach().cpu())

            module.register_forward_pre_hook(hook_fn)
            print(f"💡 已为 ReLU 层 '{name}' 注册输入捕获 Hook。")

    if not relu_input_data:
        print("❌ 未在模型中找到任何 nn.ReLU 层。请检查模型结构。")
        return

    # 3. 加载 CIFAR-10 训练数据集 (修改部分)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 将 train=False 修改为 train=True
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"🚀 开始使用 CIFAR-10 训练数据集进行前向传播以捕获所有 ReLU 层输入...")
    with torch.no_grad():
        # 将 testloader 替换为 trainloader
        for i, (inputs, labels) in enumerate(trainloader):
            _ = model(inputs)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(trainloader)} batches.")

    print("✅ 前向传播完成，所有 ReLU 层的输入数据已捕获。")

    # 4. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, 'distributions_per_channel')
    os.makedirs(images_dir, exist_ok=True)
    txt_path = os.path.join(save_dir, 'relu_input_ranges.txt')

    # 5. 可视化并保存结果
    with open(txt_path, 'w') as f_txt:
        f_txt.write("ReLU Layer Input Ranges\n")
        f_txt.write("=======================\n\n")

        for name, data_list in relu_input_data.items():
            if not data_list:
                print(f"⚠️ 警告：未捕获到 '{name}' 层的输入数据，跳过。")
                continue

            # 将所有批次的输入数据连接起来
            all_inputs = torch.cat(data_list, dim=0)
            
            # 获取数据维度：[batch_size, channels, height, width]
            num_channels = all_inputs.shape[1]
            
            # --- 写入 txt 文件 ---
            global_min = all_inputs.min().item()
            global_max = all_inputs.max().item()
            f_txt.write(f"Layer Name: {name}\n")
            f_txt.write(f"  Global Input Range: [{global_min:.6f}, {global_max:.6f}]\n")
            f_txt.write("  Per-Channel Input Ranges:\n")
            for c in range(num_channels):
                channel_data = all_inputs[:, c, :, :].numpy().flatten()
                channel_min = np.min(channel_data)
                channel_max = np.max(channel_data)
                f_txt.write(f"    Channel {c}: [{channel_min:.6f}, {channel_max:.6f}]\n")
            f_txt.write("\n")
            
            # --- 可视化：每个通道一个子图 ---
            cols = int(np.ceil(np.sqrt(num_channels)))
            rows = int(np.ceil(num_channels / cols))

            x_min_all = global_min - abs(global_min * 0.1)
            x_max_all = global_max + abs(global_max * 0.1)
            
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
            axes = axes.flatten()

            for c in range(num_channels):
                ax = axes[c]
                channel_data = all_inputs[:, c, :, :].numpy().flatten()
                ax.hist(channel_data, bins=100, color='blue', alpha=0.7)
                ax.set_title(f'Channel {c}', fontsize=8)
                ax.set_yscale('log')
                ax.tick_params(axis='both', which='major', labelsize=6)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_xlim(x_min_all, x_max_all)

            for i in range(num_channels, len(axes)):
                fig.delaxes(axes[i])
            
            plt.suptitle(f'ReLU Input Distribution for Layer: {name} (Total Channels: {num_channels})', fontsize=12)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            save_filename = os.path.join(images_dir, f'relu_input_distribution_{name.replace(".", "_")}.png')
            plt.savefig(save_filename, dpi=200)
            plt.close()
            print(f"🖼️ '{name}' 层的分通道输入分布图已保存到 {save_filename}")
    
    print(f"\n💾 所有 ReLU 层的输入范围信息已保存到 {txt_path}")
    print("\n--- 脚本执行完毕 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and visualize the input distributions of all ReLU layers in a ResNet20 model.')
    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/base_models/20_ckpt_92.23.pth",
                        help='Path to the .pth model file.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for data loading.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--save_dir', type=str, 
                        default="/home/wangtianyu/relu_finetune/hook/relu_distributions_per_channel_trainset", # 建议修改保存目录以区分
                        help='Directory to save the distribution plots and text files.')
    
    args = parser.parse_args()

    get_relu_input_distributions_per_channel(args.pth_file_path, args.batch_size, args.data_dir, args.save_dir)