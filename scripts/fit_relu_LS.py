import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
import argparse
import torchvision
import torchvision.transforms as transforms
from resnet import resnet20 # 确保你可以从 resnet.py 导入 resnet20 模型

def get_bn1_channel1_output_data(pth_file_path, batch_size, data_dir):
    """
    获取 ResNet20 模型中 bn1 层 channel 1 的输出数据。
    """
    model = resnet20()
    try:
        model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
        print(f"成功加载模型权重：{pth_file_path}")
    except Exception as e:
        print(f"加载模型权重失败：{e}")
        print("请确保你的 .pth 文件与 ResNet20 模型结构匹配。")
        return None

    model.eval()

    bn1_output_data = []

    # 注册 bn1 层的 forward hook
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name == 'bn1': # 假设bn1是模型中的顶层BN
            def hook_fn(module, input, output):
                # 捕获 bn1 层的所有输出
                bn1_output_data.append(output.detach().cpu())
            module.register_forward_hook(hook_fn)
            break # 找到 bn1 后就退出循环

    # 加载 CIFAR-10 测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"开始使用 CIFAR-10 数据集进行前向传播以捕获 bn1 输出...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            _ = model(inputs)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(testloader)} batches.")
            # 为了避免捕获过多数据导致内存问题，可以考虑只处理一部分批次
            # if i >= 50: # 例如，只处理前50个批次
            #     break

    if not bn1_output_data:
        print("警告：没有捕获到 bn1 层的输出，请检查模型结构或hook注册。")
        return None

    # 将所有批次的 bn1 输出连接起来
    all_bn1_outputs = torch.cat(bn1_output_data, dim=0) # (total_samples, num_channels, H, W)

    # 提取 bn1 的 channel 1 数据 (索引为 0)
    # 形状: (total_samples, H, W)
    channel1_data = all_bn1_outputs[:, 0, :, :].numpy()
    
    # 展平数据为一维数组
    flat_channel1_data = channel1_data.flatten()
    
    print(f"已获取 bn1 层 channel 1 的 {len(flat_channel1_data)} 个数据点。")
    return flat_channel1_data

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def polynomial_fit_relu(args):
    """
    使用多项式拟合 ReLU 函数，并可视化结果。
    """
    # 1. 获取 bn1 channel 1 的输出数据
    bn1_channel1_data = get_bn1_channel1_output_data(
        args.pth_file_path, args.batch_size, args.data_dir
    )
    if bn1_channel1_data is None:
        return

    # 确定拟合的输入范围
    # 拟合范围取 bn1 channel 1 数据的最小值和最大值
    x_min = np.min(bn1_channel1_data)
    x_max = np.max(bn1_channel1_data)
    
    # 为了更好的拟合和可视化，我们可以在整个数据范围内生成均匀分布的x值
    x_range = np.linspace(x_min, x_max, 1000)
    y_true = relu(x_range)

    print(f"拟合范围：[{x_min:.4f}, {x_max:.4f}]")
    print(f"多项式阶数：{args.poly_order}")

    # 2. 多项式拟合
    # np.polyfit(x, y, deg) 返回多项式系数，最高次幂在前
    # 为了拟合ReLU，我们需要在原始的 bn1_channel1_data 上进行采样，并计算其对应的ReLU真值
    # 这里我们简化，直接使用 x_range 作为拟合的输入，虽然更精确的做法是使用原始数据点的 x 和对应的 ReLU(x)
    # 但由于我们的目标是拟合ReLU函数本身，而不是原始数据点，所以这种方式是合理的。
    
    # 随机采样一部分数据点进行拟合，以防止数据量过大影响性能或拟合效果
    # 更好的做法是在整个输入范围内均匀采样，并使用这些采样点及其对应的ReLU值进行拟合
    # 再次强调，这里拟合的是 'x' 到 'relu(x)' 的关系，而不是 '原始数据点' 到 'relu(原始数据点)'
    # 所以我们应该用一个均匀分布的x_train_fit来代表输入范围
    
    # 重新定义拟合用的X和Y
    # 在bn1_channel1_data的范围内，生成用于拟合的X和对应的ReLU Y
    # 为了避免在大量相同值上拟合，我们使用更均匀分布的采样点
    x_for_fit = np.linspace(x_min, x_max, 500) # 500个均匀分布的采样点
    y_for_fit = relu(x_for_fit)

    # 避免阶数过高导致警告或过拟合
    if args.poly_order >= len(x_for_fit):
        print(f"警告：多项式阶数 ({args.poly_order}) 大于或等于拟合点的数量 ({len(x_for_fit)})。")
        print("这可能导致过拟合或数值不稳定。将阶数调整为拟合点数量减1。")
        args.poly_order = len(x_for_fit) - 1
        if args.poly_order < 0:
            print("没有足够的点进行拟合。")
            return

    poly_coeffs = np.polyfit(x_for_fit, y_for_fit, args.poly_order)
    poly_func = np.poly1d(poly_coeffs) # 创建多项式函数

    # 3. 计算 MSE 误差
    y_predicted = poly_func(x_range)
    mse = np.mean((y_true - y_predicted)**2)
    print(f"多项式拟合的 MSE 误差: {mse:.6f}")

    # 4. 可视化
    plt.figure(figsize=(10, 6))
    
    # 绘制原始 bn1 channel 1 数据的直方图（背景）
    plt.hist(bn1_channel1_data, bins=100, density=True, alpha=0.3, color='lightgray', label='BN1 Channel 1 Output Data Distribution')

    # 绘制 ReLU 真值
    plt.plot(x_range, y_true, label='True ReLU Function', color='red', linestyle='--', linewidth=2)

    # 绘制拟合多项式
    plt.plot(x_range, y_predicted, label=f'Polynomial Fit (Order {args.poly_order})', color='blue', linewidth=2)

    plt.title(f'ReLU Function Polynomial Approximation (Order {args.poly_order})\nMSE: {mse:.6f}')
    plt.xlabel('Input Value')
    plt.ylabel('Output Value / Density')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    save_filename = os.path.join(args.save_dir, f'relu_polynomial_fit_order_{args.poly_order}.jpg')
    plt.savefig(save_filename, dpi=200)
    plt.close()
    print(f"拟合结果图已保存到 {save_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit ReLU function with a polynomial.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/my_resnet20/base_models/20_ckpt_91.39_usp0.75.pth",
                        help='Path to the .pth model file (used to get BN1 Channel 1 output range).')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for data loading (used for getting BN1 Channel 1 output range).')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--save_dir', type=str, 
                        default="/home/wangtianyu/my_resnet20/hook/appReLU",
                        help='Directory to save the polynomial fit plot.')
    parser.add_argument('--poly_order', type=int, default=7,
                        help='Order of the polynomial to fit the ReLU function.')

    args = parser.parse_args()

    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 运行拟合和可视化
    polynomial_fit_relu(args)

    print("\n多项式拟合脚本执行完毕。")