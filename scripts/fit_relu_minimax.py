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

# 导入 scipy 的线性规划求解器
from scipy.optimize import linprog

# （get_bn1_channel1_output_data 和 relu 函数与之前保持不变）
def get_bn1_channel1_output_data(pth_file_path, batch_size, data_dir):
    """
    获取 ResNet20 模型中 bn1 层 channel 1 的输出数据。
    （与之前代码相同，省略具体实现）
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

    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name == 'bn1':
            def hook_fn(module, input, output):
                bn1_output_data.append(output.detach().cpu())
            module.register_forward_hook(hook_fn)
            break

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
            # if i >= 50: # 例如，只处理前50个批次
            #     break

    if not bn1_output_data:
        print("警告：没有捕获到 bn1 层的输出，请检查模型结构或hook注册。")
        return None

    all_bn1_outputs = torch.cat(bn1_output_data, dim=0)
    channel1_data = all_bn1_outputs[:, 0, :, :].numpy()
    flat_channel1_data = channel1_data.flatten()
    
    print(f"已获取 bn1 层 channel 1 的 {len(flat_channel1_data)} 个数据点。")
    return flat_channel1_data

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def polynomial_fit_relu_minimax(args):
    """
    使用多项式和 Minimax 优化目标拟合 ReLU 函数，并可视化结果。
    """
    bn1_channel1_data = get_bn1_channel1_output_data(
        args.pth_file_path, args.batch_size, args.data_dir
    )
    if bn1_channel1_data is None:
        return

    x_min = np.min(bn1_channel1_data)
    x_max = np.max(bn1_channel1_data)
    
    # 拟合用的 X 和 Y 数据点。在 Minimax 拟合中，我们通常需要更多的点来确保最大误差被准确捕捉。
    # 为了简化，我们仍然在均匀分布的 x 值上拟合 ReLU。
    x_for_fit = np.linspace(x_min, x_max, args.num_fit_points) # 控制拟合点的数量
    y_for_fit = relu(x_for_fit)

    num_coeffs = args.poly_order + 1 # 多项式系数的数量 (从 x^0 到 x^order)
    num_points = len(x_for_fit)

    # 构建线性规划问题
    # 目标函数：最小化 E (max_error)
    # 变量是：[c_n, c_{n-1}, ..., c_0, E]
    # c_i 是多项式系数，E 是最大绝对误差
    
    # c (目标函数系数): 对于 E 设为 1，对于其他多项式系数设为 0
    c = np.zeros(num_coeffs + 1)
    c[-1] = 1 # 最后一个变量是 E

    # A_ub 和 b_ub (不等式约束):
    # f(x_i) - P(x_i) <= E  => P(x_i) + E >= f(x_i) => -P(x_i) - E <= -f(x_i)
    # P(x_i) - f(x_i) <= E  => P(x_i) - E <= f(x_i)

    # 展开 P(x_i) = c_n x_i^n + c_{n-1} x_i^{n-1} + ... + c_0
    
    # 构建约束矩阵 A_ub 和 b_ub
    A_ub = np.zeros((2 * num_points, num_coeffs + 1))
    b_ub = np.zeros(2 * num_points)

    for i in range(num_points):
        x_val = x_for_fit[i]
        y_val = y_for_fit[i]

        # 约束 1: P(x_i) - E <= f(x_i)
        # c_n x_i^n + ... + c_0 - E <= y_val
        for j in range(num_coeffs): # 对应系数 c_n 到 c_0
            A_ub[i, j] = x_val**(args.poly_order - j)
        A_ub[i, -1] = -1 # 对应 -E
        b_ub[i] = y_val

        # 约束 2: -P(x_i) - E <= -f(x_i)
        # -(c_n x_i^n + ... + c_0) - E <= -y_val
        for j in range(num_coeffs):
            A_ub[num_points + i, j] = -x_val**(args.poly_order - j)
        A_ub[num_points + i, -1] = -1 # 对应 -E
        b_ub[num_points + i] = -y_val
    
    # E >= 0 的边界约束，由 bounds 参数处理

    # 变量边界：多项式系数可以是任何实数 (None, None)，E 必须 >= 0 (0, None)
    bounds = [(None, None)] * num_coeffs + [(0, None)] # for c_i, and for E

    print(f"开始进行 Minimax 多项式拟合 (阶数: {args.poly_order})...")
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs') # 'highs' 是推荐的默认方法

    if result.success:
        poly_coeffs_minimax = result.x[:-1] # 最后一个是 E，我们只需要系数
        max_abs_error = result.x[-1]
        
        poly_func_minimax = np.poly1d(poly_coeffs_minimax) # 创建多项式函数

        # 评估拟合效果
        x_range_plot = np.linspace(x_min, x_max, 1000)
        y_true_plot = relu(x_range_plot)
        y_predicted_minimax = poly_func_minimax(x_range_plot)
        
        # 计算 MSE (虽然优化目标不是 MSE，但我们仍可以计算它作为参考)
        mse_minimax = np.mean((y_true_plot - y_predicted_minimax)**2)

        print(f"Minimax 拟合成功！")
        print(f"最大绝对误差 (Max Abs Error): {max_abs_error:.6f}")
        print(f"参考 MSE 误差: {mse_minimax:.6f}")

        # 5. 可视化
        plt.figure(figsize=(10, 6))
        
        plt.hist(bn1_channel1_data, bins=100, density=True, alpha=0.3, color='lightgray', label='BN1 Channel 1 Output Data Distribution')
        plt.plot(x_range_plot, y_true_plot, label='True ReLU Function', color='red', linestyle='--', linewidth=2)
        plt.plot(x_range_plot, y_predicted_minimax, label=f'Minimax Poly Fit (Order {args.poly_order})', color='green', linewidth=2)

        plt.title(f'ReLU Function Minimax Polynomial Approximation (Order {args.poly_order})\nMax Abs Error: {max_abs_error:.6f} | Ref MSE: {mse_minimax:.6f}')
        plt.xlabel('Input Value')
        plt.ylabel('Output Value / Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        save_filename = os.path.join(args.save_dir, f'relu_minimax_fit_order_{args.poly_order}.jpg')
        plt.savefig(save_filename, dpi=200)
        plt.close()
        print(f"Minimax 拟合结果图已保存到 {save_filename}")

    else:
        print(f"Minimax 拟合失败：{result.message}")
        print("请尝试调整多项式阶数或拟合点数量。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit ReLU function with a polynomial using Minimax Composition.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/my_resnet20/base_models/20_ckpt_91.39_usp0.75.pth",
                        help='Path to the .pth model file (used to get BN1 Channel 1 output range).')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for data loading (used for getting BN1 Channel 1 output range).')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--save_dir', type=str, 
                        default="/home/wangtianyu/my_resnet20/hook/appReLU_minimax",
                        help='Directory to save the polynomial fit plot.')
    parser.add_argument('--poly_order', type=int, default=7,
                        help='Order of the polynomial to fit the ReLU function.')
    parser.add_argument('--num_fit_points', type=int, default=500,
                        help='Number of evenly spaced points to use for fitting the ReLU function.')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    polynomial_fit_relu_minimax(args)

    print("\nMinimax 多项式拟合脚本执行完毕。")