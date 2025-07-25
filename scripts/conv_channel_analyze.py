import torch
import torch.nn as nn
import torch.nn.functional as F # 用于交叉熵损失
import numpy as np
import os
import argparse
import torchvision
import torchvision.transforms as transforms
from datetime import datetime # 用于记录时间

# 假设你的 ResNet20 模型定义在 resnet.py 文件中
# 请确保 resnet.py 文件在当前脚本可导入的路径下
from resnet import resnet20 

def analyze_conv_channel_influence(args):
    """
    计算并记录每个卷积层输入通道的绝对平均梯度，以衡量其影响。

    Args:
        args (argparse.Namespace): 包含所有配置参数的对象。
    """
    # 确保日志文件所在的目录存在
    log_dir = os.path.dirname(args.log_file_path)
    if log_dir: # 如果log_file_path包含目录，则创建
        os.makedirs(log_dir, exist_ok=True)
    
    # 1. 加载模型
    model = resnet20()
    try:
        model.load_state_dict(torch.load(args.pth_file_path, map_location=torch.device('cpu')))
        print(f"成功加载模型权重：{args.pth_file_path}")
    except Exception as e:
        print(f"加载模型权重失败：{e}")
        print("请确保你的 .pth 文件与 ResNet20 模型结构匹配。")
        return

    model.eval() # 设置为评估模式，这会使 BN 层使用训练时的均值和方差，并禁用 Dropout。
                 # 但我们仍需要梯度，所以后续会确保输入张量的 requires_grad=True。

    # 2. 梯度积累容器
    # 存储每个卷积层输入通道的绝对梯度列表
    # 键是卷积层名称 (str)，值是梯度张量列表 (list of torch.Tensor)
    conv_input_gradients = {}
    
    # 存储每个卷积层输入通道的形状 (C, H, W)，用于日志输出和验证
    conv_layer_input_shapes = {}

    # 存储注册的 hook handler，方便之后移除，防止内存泄漏
    hook_handlers = []

    # 3. 注册 hook
    # 遍历模型的所有模块，找到 nn.Conv2d 层
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 初始化该卷积层的梯度列表
            conv_input_gradients[name] = []

            # 定义一个闭包函数，以便 forward_hook_fn 能访问 layer_name
            def make_forward_hook(layer_name):
                def forward_hook_fn(module, input_tensors, output_tensor):
                    # input_tensors 是一个元组，通常 input_tensors[0] 是实际输入张量
                    input_tensor = input_tensors[0]
                    
                    # 确保这个张量需要梯度，否则无法注册 backward hook
                    if input_tensor.requires_grad:
                        # 注册 backward hook 到输入张量
                        # 这个 hook 会在 loss.backward() 被调用时触发，并接收该张量的梯度
                        def backward_hook_fn(grad):
                            # grad 是张量的梯度。我们取其绝对值并移动到 CPU，然后添加到列表中
                            # 每个 batch 的梯度都会被累积到这里
                            conv_input_gradients[layer_name].append(grad.abs().cpu())
                        
                        # 注册 backward hook 并保存其 handler
                        hook_handle = input_tensor.register_hook(backward_hook_fn)
                        hook_handlers.append(hook_handle)
                        
                        # 记录输入形状 (C, H, W)，假设所有批次输入形状相同，只记录一次
                        if layer_name not in conv_layer_input_shapes:
                            conv_layer_input_shapes[layer_name] = input_tensor.shape[1:] # (Channels, Height, Width)
                return forward_hook_fn
            
            # 注册 forward hook 到当前的卷积模块
            hook_handle = module.register_forward_hook(make_forward_hook(name))
            hook_handlers.append(hook_handle)

    # 4. 加载 CIFAR-10 数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"开始使用 CIFAR-10 数据集计算卷积层输入通道的梯度...")
    print(f"总批次数量：{len(testloader)}")

    # 5. 前向和反向传播以积累梯度
    total_samples_processed = 0
    # 关键修复：移除 with torch.no_grad(): 块，以确保计算图被正确构建，从而可以计算梯度。
    for i, (inputs, labels) in enumerate(testloader):
        # 必须为模型的初始输入手动设置 requires_grad=True，以便 PyTorch 构建计算图并计算梯度
        inputs.requires_grad_(True) 
        
        outputs = model(inputs)
        # 使用交叉熵损失作为梯度计算的起点。
        loss = F.cross_entropy(outputs, labels)
        
        # 执行反向传播，这将触发所有注册的 backward hooks
        loss.backward()

        total_samples_processed += inputs.size(0)

        if (i + 1) % 100 == 0:
            print(f"  已处理 {i + 1}/{len(testloader)} 批次。总样本数: {total_samples_processed}")
        
        # # 调试/测试用：只处理一小部分数据，以快速验证脚本功能
        # if i >= 10:
        #     print("DEBUG: 只处理了前10个批次.")
        #     break

    # 6. 计算每个卷积层输入通道的绝对平均梯度并写入日志文件
    log_messages = []
    log_messages.append(f"--- 卷积层输入通道绝对平均梯度分析 ({os.path.basename(args.pth_file_path)}) ---")
    log_messages.append(f"分析日期和时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_messages.append(f"PyTorch 版本: {torch.__version__}")
    log_messages.append(f"数据集: CIFAR-10 测试集 ({total_samples_processed} 样本)")
    log_messages.append(f"Batch Size: {args.batch_size}")
    log_messages.append("-" * 80)

    # 遍历收集到的每个卷积层的梯度数据
    for conv_layer_name, grads_list in conv_input_gradients.items():
        if not grads_list:
            log_messages.append(f"\n警告: 卷积层 '{conv_layer_name}' 没有捕获到输入梯度。可能未被前向传播激活或 requires_grad 未设置。")
            continue

        # 将所有批次（从 hook 收集的）的梯度张量连接起来
        # grads_list 中的每个张量形状通常是 (batch_size, C, H, W)
        all_grads_for_layer = torch.cat(grads_list, dim=0)
        
        # 计算每个通道的绝对平均梯度
        # dim=(0, 2, 3) 意味着在 batch 维度、高度维度和宽度维度上求平均，只保留通道维度
        avg_abs_grad_per_channel = all_grads_for_layer.mean(dim=(0, 2, 3))

        log_messages.append(f"\n卷积层: {conv_layer_name}")
        log_messages.append(f"  输入形状 (C, H, W): {conv_layer_input_shapes.get(conv_layer_name, '未知')}")
        log_messages.append(f"  输入通道数: {avg_abs_grad_per_channel.shape[0]}")
        
        # 按梯度值从高到低排序，以便更容易看出重要通道
        sorted_channel_indices = torch.argsort(avg_abs_grad_per_channel, descending=True)
        
        log_messages.append("  通道绝对平均梯度 (降序):")
        for i in sorted_channel_indices:
            log_messages.append(f"    通道 {i.item():3d}: {avg_abs_grad_per_channel[i].item():.8f}")
    
    # 将所有日志信息写入文件
    with open(args.log_file_path, 'w') as f:
        for msg in log_messages:
            f.write(msg + '\n')
    
    print(f"\n分析结果已保存到日志文件: {args.log_file_path}")

    # 7. 清理：移除所有注册的 hook，防止内存泄漏或干扰后续操作
    for handle in hook_handlers:
        handle.remove()
    print("所有注册的 hooks 已移除。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze influence of input channels in convolutional layers using absolute average gradients.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/my_resnet20/base_models/20_ckpt_91.39_usp0.75.pth",
                        help='Path to the .pth model file.')
    parser.add_argument('--batch_size', type=int, default=512, # 可以根据你的内存情况调整
                        help='Batch size for data loading.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--log_file_path', type=str, 
                        default="/home/wangtianyu/my_resnet20/hook/conv_analysis/conv_channel_influence.log", # 默认日志文件路径
                        help='Path to the output log file.')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading. Set to 0 for single-process loading (for debugging).')
    
    args = parser.parse_args()

    analyze_conv_channel_influence(args)
    print("\n脚本执行完毕。")