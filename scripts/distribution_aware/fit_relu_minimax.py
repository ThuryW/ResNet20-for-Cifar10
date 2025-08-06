import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import json
from collections import OrderedDict
from scipy.optimize import minimize
import torchvision
import torchvision.transforms as transforms
from resnet import resnet20 # 确保你可以从 resnet.py 导入 resnet20 模型

# --- (以下是获取 ReLU 输入数据的函数，修改以返回原始 tensor 和展平数据) ---
def get_relu_input_data_for_layer(model, relu_layer_name, batch_size, data_dir):
    """
    获取指定 ReLU 层的输入数据。
    返回展平的 NumPy 数组（用于拟合）和原始的 PyTorch Tensor 列表（用于通道分布）。
    """
    relu_input_data_list = []

    found_relu = False
    # 注册 forward_pre_hook 来捕获 ReLU 层的输入
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and name == relu_layer_name:
            def hook_fn(module, input):
                relu_input_data_list.append(input[0].detach().cpu())
            module.register_forward_pre_hook(hook_fn)
            found_relu = True
            # print(f"💡 已为 ReLU 层 '{relu_layer_name}' 注册输入捕获 Hook。")
            break
    
    if not found_relu:
        print(f"❌ 未在模型中找到指定的 ReLU 层 '{relu_layer_name}'。")
        return None, None

    # 加载 CIFAR-10 测试数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # print(f"🚀 开始使用 CIFAR-10 数据集进行前向传播以捕获 '{relu_layer_name}' 输入...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            _ = model(inputs)
            # if (i + 1) % 100 == 0:
            #     print(f"  Processed {i + 1}/{len(testloader)} batches.")
            # 为避免内存问题，可以限制处理的批次数量
            # if i >= 10: break

    if not relu_input_data_list:
        print(f"⚠️ 警告：没有捕获到 '{relu_layer_name}' 层的输入。")
        return None, None

    # 将所有批次的 ReLU 输入连接起来
    all_relu_inputs_tensor = torch.cat(relu_input_data_list, dim=0) # 原始形状 (N, C, H, W)
    all_relu_inputs_flat = all_relu_inputs_tensor.numpy().flatten() # 展平为一维数组用于拟合

    return all_relu_inputs_flat, all_relu_inputs_tensor

def relu_true(x):
    """标准的 ReLU 激活函数"""
    return np.maximum(0, x)

def poly_eval(x, coeffs):
    """
    使用 Horner 法则计算多项式值。
    coeffs: 从最高次幂到最低次幂的系数列表或 NumPy 数组 [c_n, c_{n-1}, ..., c_0]
    """
    if len(coeffs) == 0:
        return np.zeros_like(x)
    
    result = coeffs[0] * x
    for i in range(1, len(coeffs) - 1):
        result = result + coeffs[i]
        result = result * x
    result = result + coeffs[-1]
    return result

def f1_f2_composite(x, coeffs_f1, coeffs_f2):
    """
    计算复合函数 f1(f2(x)) 的值。
    """
    y_intermediate = poly_eval(x, coeffs_f2)
    return poly_eval(y_intermediate, coeffs_f1)

def objective_function(packed_coeffs, x_sample, y_true, deg1, deg2):
    """
    优化目标函数：计算复合函数与真实 ReLU 之间的最大绝对误差 (L-infinity norm)。
    """
    num_coeffs_f1 = deg1 + 1
    # num_coeffs_f2 = deg2 + 1 # 未直接使用，但用于理解
    
    coeffs_f1 = packed_coeffs[:num_coeffs_f1]
    coeffs_f2 = packed_coeffs[num_coeffs_f1:]

    y_predicted = f1_f2_composite(x_sample, coeffs_f1, coeffs_f2)
    
    max_abs_error = np.max(np.abs(y_predicted - y_true))
    return max_abs_error

def fit_single_relu_layer(model, relu_layer_name, args, global_save_dir):
    """
    对单个 ReLU 层进行 Minimax 复合多项式拟合，并保存结果。
    """
    print(f"\n--- 正在处理 ReLU 层: '{relu_layer_name}' ---")
    
    # 1. 获取 ReLU 输入数据
    all_relu_inputs_flat, all_relu_inputs_tensor = get_relu_input_data_for_layer(
        model, relu_layer_name, args.batch_size, args.data_dir
    )
    
    if all_relu_inputs_flat is None:
        return {
            'status': 'skipped',
            'message': 'Failed to capture input data or layer not found.',
            'relu_layer_name': relu_layer_name
        }

    # 确定拟合的输入范围：从数据中观察到的最小值到最大值
    x_min = np.min(all_relu_inputs_flat)
    x_max = np.max(all_relu_inputs_flat)
    
    # 在这个范围内均匀采样用于拟合的点
    x_range_for_fit = np.linspace(x_min, x_max, args.num_fit_points)
    y_true_for_fit = relu_true(x_range_for_fit)

    print(f"  📊 拟合范围：[{x_min:.4f}, {x_max:.4f}]")
    print(f"  📏 f1 多项式阶数 (d1)：{args.degree_f1}")
    print(f"  📏 f2 多项式阶数 (d2)：{args.degree_f2}")
    print(f"  🧪 复合多项式 f1(f2(x)) 理论最高阶数：{args.degree_f1 * args.degree_f2}")
    print(f"  🧮 拟合点数量：{args.num_fit_points}")
    print(f"  🔍 优化器：{args.optimizer_method}")

    # 检查多项式阶数是否合理
    total_coeffs = (args.degree_f1 + 1) + (args.degree_f2 + 1)
    if total_coeffs > args.num_fit_points:
        print(f"❌ 错误：多项式系数总数 ({total_coeffs}) 大于拟合点数量 ({args.num_fit_points})。")
        print("  请增加 --num_fit_points 或降低多项式阶数。跳过此层。")
        return {
            'status': 'failed',
            'message': 'Insufficient fit points for polynomial degrees.',
            'relu_layer_name': relu_layer_name,
            'degree_f1': args.degree_f1,
            'degree_f2': args.degree_f2,
            'x_min': float(x_min),
            'x_max': float(x_max)
        }

    # 2. 初始化 f1 和 f2 的系数
    # 使用简单的线性拟合作为初始猜测
    initial_coeffs_f2 = np.polyfit(x_range_for_fit, x_range_for_fit, args.degree_f2)
    initial_y_for_f1 = poly_eval(x_range_for_fit, initial_coeffs_f2)
    initial_coeffs_f1 = np.polyfit(initial_y_for_f1, y_true_for_fit, args.degree_f1)

    initial_packed_coeffs = np.concatenate((initial_coeffs_f1, initial_coeffs_f2))

    # print(f"  ✨ 初始拟合的最大绝对误差：{objective_function(initial_packed_coeffs, x_range_for_fit, y_true_for_fit, args.degree_f1, args.degree_f2):.6f}")

    # 3. 执行 Minimax 优化
    print(f"  🚀 开始 Minimax 拟合优化 '{relu_layer_name}'...")
    result = minimize(
        objective_function,
        initial_packed_coeffs,
        args=(x_range_for_fit, y_true_for_fit, args.degree_f1, args.degree_f2),
        method=args.optimizer_method,
        options={'maxiter': args.max_iterations, 'disp': True} # disp=True 打印优化过程
    )

    if result.success:
        print(f"  🎉 优化成功！")
        optimized_packed_coeffs = result.x
        final_max_error = result.fun
    else:
        print(f"  ❌ 优化失败或未收敛：{result.message}")
        optimized_packed_coeffs = initial_packed_coeffs # 失败时使用初始系数
        final_max_error = objective_function(initial_packed_coeffs, x_range_for_fit, y_true_for_fit, args.degree_f1, args.degree_f2)

    num_coeffs_f1 = args.degree_f1 + 1
    optimized_coeffs_f1 = optimized_packed_coeffs[:num_coeffs_f1]
    optimized_coeffs_f2 = optimized_packed_coeffs[num_coeffs_f1:]

    y_predicted_final = f1_f2_composite(x_range_for_fit, optimized_coeffs_f1, optimized_coeffs_f2)
    
    # 计算 MSE 误差
    mse = np.mean((y_true_for_fit - y_predicted_final)**2)

    print(f"  🎯 最终拟合的最大绝对误差 (Minimax Error): {final_max_error:.6f}")
    print(f"  📈 最终拟合的 MSE 误差: {mse:.6f}")

    # 4. 可视化结果 (针对每个通道的分布)
    layer_save_dir = os.path.join(global_save_dir, relu_layer_name.replace(".", "_"))
    os.makedirs(layer_save_dir, exist_ok=True)

    # all_relu_inputs_tensor 的形状: (N, C, H, W)
    # 展平 HxW 维度，以便按通道查看分布 (C, N*H*W)
    channel_data_for_plot = all_relu_inputs_tensor.permute(1, 0, 2, 3).reshape(all_relu_inputs_tensor.shape[1], -1).numpy()
    num_channels = channel_data_for_plot.shape[0]

    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if num_channels > 1 else np.array([axes])

    for i in range(num_channels):
        ax = axes[i]
        ax.hist(channel_data_for_plot[i], bins=50, density=True, alpha=0.7, color='skyblue')
        # 在每个通道的直方图上绘制同一条拟合曲线
        ax.plot(x_range_for_fit, y_predicted_final, color='red', linewidth=2, label=f'Composite Fit')
        ax.set_title(f'Channel {i+1}')
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)
        # if i == 0: ax.legend() # 只在第一个子图显示图例

    for i in range(num_channels, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'ReLU Layer: {relu_layer_name}\nMinimax Composite Fit (d1={args.degree_f1}, d2={args.degree_f2})\nMax Abs Error: {final_max_error:.6f}, MSE: {mse:.6f}', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    save_filename = os.path.join(layer_save_dir, f'{relu_layer_name.replace(".", "_")}_composite_fit_channels.jpg')
    plt.savefig(save_filename, dpi=200)
    plt.close(fig)
    print(f"  🖼️ 拟合结果图已保存到 {save_filename}")

    # 5. 返回拟合结果
    return {
        'status': 'success' if result.success else 'failed',
        'message': result.message,
        'relu_layer_name': relu_layer_name,
        'degree_f1': args.degree_f1,
        'degree_f2': args.degree_f2,
        'f1_coeffs': optimized_coeffs_f1.tolist(),
        'f2_coeffs': optimized_coeffs_f2.tolist(),
        'x_min': float(x_min),
        'x_max': float(x_max),
        'final_max_error': float(final_max_error),
        'mse': float(mse),
        'optimization_success': bool(result.success)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Minimax fit ReLU function with a composite polynomial f1(f2(x)) for all ReLU layers.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/base_models/20_ckpt_92.23.pth",
                        help='Path to the .pth model file.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for data loading.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--save_dir', type=str, 
                        default="/home/wangtianyu/relu_finetune/hook/minimax_composite_relu_all_layers",
                        help='Root directory to save fit plots and coefficients.')
    
    parser.add_argument('--degree_f1', type=int, default=31,
                        help='Degree of the first polynomial f1.')
    parser.add_argument('--degree_f2', type=int, default=31,
                        help='Degree of the second polynomial f2.')
    
    parser.add_argument('--num_fit_points', type=int, default=5000,
                        help='Number of uniformly sampled points in the range for fitting.')
    parser.add_argument('--optimizer_method', type=str, default='Nelder-Mead',
                        help='Optimization method for scipy.optimize.minimize. '
                             'Options: "Nelder-Mead", "Powell", "SLSQP", "L-BFGS-B" (try others if one fails).')
    parser.add_argument('--max_iterations', type=int, default=20000,
                        help='Maximum number of iterations for the optimizer.')


    args = parser.parse_args()

    # 确保主保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载模型 (只需加载一次)
    model = resnet20()
    try:
        checkpoint = torch.load(args.pth_file_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✨ 成功加载模型权重：{args.pth_file_path}")
    except Exception as e:
        print(f"❌ 加载模型权重失败：{e}")
        print("请确保你的 .pth 文件与 ResNet20 模型结构匹配。")
        exit()
    
    model.eval() # 设置为评估模式
    
    all_relu_fit_results = OrderedDict()
    
    # 2. 遍历所有 ReLU 层并进行拟合
    relu_layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            relu_layer_names.append(name)
    
    if not relu_layer_names:
        print("未在模型中找到任何 nn.ReLU 层。")
        exit()

    print(f"\n模型中找到 {len(relu_layer_names)} 个 ReLU 层。开始逐层拟合...")

    # 每次循环，需要重新注册 Hook，因为 Hook 会在捕获后失效或累积
    # 因此，我们为每个层创建一个临时的模型实例或重新加载模型，以确保 Hook 的独立性
    # 更高效的方法是，在 get_relu_input_data_for_layer 内部注册并移除 Hook，但PyTorch Hook生命周期管理复杂
    # 最简单的做法是每次获取数据时都重新加载模型，以确保模型状态干净
    
    # 注意：为了避免多次加载模型，我们在这里使用一个“clean_model”的概念
    # 但是 hook 机制会随着 forward pass 触发。为了确保每次 hook 捕获的是当前层的输入，
    # 且不被之前的 hook 干扰，需要重新构建模型或手动移除 hook。
    # 最简单的做法是让 get_relu_input_data_for_layer 函数在获取数据后，
    # 移除其注册的 hook，以避免对后续层的影响。但 PyTorch 默认不会自动移除。
    # 因此，目前最稳妥的方法是，每次调用 get_relu_input_data_for_layer 时，都重新加载一个模型实例
    # 这样可以确保每个hook都是独立的，不会互相干扰。但会导致加载模型多次。
    # 考虑到数据捕获只在拟合开始时执行一次，其性能开销通常小于优化本身。
    
    # 修改：get_relu_input_data_for_layer 将接收 pth_file_path，并在内部加载模型以确保独立性
    # 这样，主循环中的 model 实例可以保持不变，用于 named_modules 遍历
    # 而每个拟合操作的数据捕获是独立的

    # 将 get_relu_input_data_for_layer 的 model 参数改为 pth_file_path
    # 并让它在内部加载模型
    def get_relu_input_data_for_layer_modified(pth_file_path, relu_layer_name, batch_size, data_dir):
        temp_model = resnet20()
        try:
            checkpoint = torch.load(pth_file_path, map_location=torch.device('cpu'))
            state_dict = checkpoint.get('net', checkpoint)
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            temp_model.load_state_dict(new_state_dict)
            temp_model.eval()
        except Exception as e:
            print(f"❌ 临时模型加载失败：{e}")
            return None, None

        # 注册 hook 并运行数据捕获
        relu_input_data_list = []
        hook_handle = None
        for name, module in temp_model.named_modules():
            if isinstance(module, nn.ReLU) and name == relu_layer_name:
                def hook_fn_local(module, input):
                    relu_input_data_list.append(input[0].detach().cpu())
                hook_handle = module.register_forward_pre_hook(hook_fn_local)
                break
        
        if hook_handle is None:
             print(f"❌ 未能为 '{relu_layer_name}' 注册 Hook。")
             return None, None

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(testloader):
                _ = temp_model(inputs)
                # if i >= 10: break # 限制批次，加快数据捕获

        if hook_handle:
            hook_handle.remove() # 移除 hook 以防止干扰

        if not relu_input_data_list:
            print(f"⚠️ 警告：没有捕获到 '{relu_layer_name}' 层的输入。")
            return None, None

        all_relu_inputs_tensor = torch.cat(relu_input_data_list, dim=0)
        all_relu_inputs_flat = all_relu_inputs_tensor.numpy().flatten()
        
        return all_relu_inputs_flat, all_relu_inputs_tensor

    # 更新 fit_single_relu_layer 函数的调用，传递 pth_file_path
    def fit_single_relu_layer_updated(pth_file_path, relu_layer_name, args, global_save_dir):
        print(f"\n--- 正在处理 ReLU 层: '{relu_layer_name}' ---")
        
        # 1. 获取 ReLU 输入数据
        all_relu_inputs_flat, all_relu_inputs_tensor = get_relu_input_data_for_layer_modified(
            pth_file_path, relu_layer_name, args.batch_size, args.data_dir
        )
        
        if all_relu_inputs_flat is None:
            return {
                'status': 'skipped',
                'message': 'Failed to capture input data or layer not found.',
                'relu_layer_name': relu_layer_name
            }

        x_min = np.min(all_relu_inputs_flat)
        x_max = np.max(all_relu_inputs_flat)
        x_range_for_fit = np.linspace(x_min, x_max, args.num_fit_points)
        y_true_for_fit = relu_true(x_range_for_fit)

        print(f"  📊 拟合范围：[{x_min:.4f}, {x_max:.4f}]")
        print(f"  📏 f1 多项式阶数 (d1)：{args.degree_f1}")
        print(f"  📏 f2 多项式阶数 (d2)：{args.degree_f2}")
        print(f"  🧪 复合多项式 f1(f2(x)) 理论最高阶数：{args.degree_f1 * args.degree_f2}")
        print(f"  🧮 拟合点数量：{args.num_fit_points}")
        print(f"  🔍 优化器：{args.optimizer_method}")

        total_coeffs = (args.degree_f1 + 1) + (args.degree_f2 + 1)
        if total_coeffs > args.num_fit_points:
            print(f"❌ 错误：多项式系数总数 ({total_coeffs}) 大于拟合点数量 ({args.num_fit_points})。")
            print("  请增加 --num_fit_points 或降低多项式阶数。跳过此层。")
            return {
                'status': 'failed',
                'message': 'Insufficient fit points for polynomial degrees.',
                'relu_layer_name': relu_layer_name,
                'degree_f1': args.degree_f1,
                'degree_f2': args.degree_f2,
                'x_min': float(x_min),
                'x_max': float(x_max)
            }

        initial_coeffs_f2 = np.polyfit(x_range_for_fit, x_range_for_fit, args.degree_f2)
        initial_y_for_f1 = poly_eval(x_range_for_fit, initial_coeffs_f2)
        initial_coeffs_f1 = np.polyfit(initial_y_for_f1, y_true_for_fit, args.degree_f1)
        initial_packed_coeffs = np.concatenate((initial_coeffs_f1, initial_coeffs_f2))

        print(f"  🚀 开始 Minimax 拟合优化 '{relu_layer_name}'...")
        result = minimize(
            objective_function,
            initial_packed_coeffs,
            args=(x_range_for_fit, y_true_for_fit, args.degree_f1, args.degree_f2),
            method=args.optimizer_method,
            options={'maxiter': args.max_iterations, 'disp': False} # disp=True 打印优化过程
        )

        if result.success:
            print(f"  🎉 优化成功！")
            optimized_packed_coeffs = result.x
            final_max_error = result.fun
        else:
            print(f"  ❌ 优化失败或未收敛：{result.message}")
            optimized_packed_coeffs = initial_packed_coeffs
            final_max_error = objective_function(initial_packed_coeffs, x_range_for_fit, y_true_for_fit, args.degree_f1, args.degree_f2)

        num_coeffs_f1 = args.degree_f1 + 1
        optimized_coeffs_f1 = optimized_packed_coeffs[:num_coeffs_f1]
        optimized_coeffs_f2 = optimized_packed_coeffs[num_coeffs_f1:]

        y_predicted_final = f1_f2_composite(x_range_for_fit, optimized_coeffs_f1, optimized_coeffs_f2)
        mse = np.mean((y_true_for_fit - y_predicted_final)**2)

        print(f"  🎯 最终拟合的最大绝对误差 (Minimax Error): {final_max_error:.6f}")
        print(f"  📈 最终拟合的 MSE 误差: {mse:.6f}")

        layer_save_dir = os.path.join(global_save_dir, relu_layer_name.replace(".", "_"))
        os.makedirs(layer_save_dir, exist_ok=True)

        channel_data_for_plot = all_relu_inputs_tensor.permute(1, 0, 2, 3).reshape(all_relu_inputs_tensor.shape[1], -1).numpy()
        num_channels = channel_data_for_plot.shape[0]

        cols = int(np.ceil(np.sqrt(num_channels)))
        rows = int(np.ceil(num_channels / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten() if num_channels > 1 else np.array([axes])

        for i in range(num_channels):
            ax = axes[i]
            ax.hist(channel_data_for_plot[i], bins=50, density=True, alpha=0.7, color='skyblue')
            ax.plot(x_range_for_fit, y_predicted_final, color='red', linewidth=2, label=f'Composite Fit')
            ax.set_title(f'Channel {i+1}')
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

        for i in range(num_channels, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'ReLU Layer: {relu_layer_name}\nMinimax Composite Fit (d1={args.degree_f1}, d2={args.degree_f2})\nMax Abs Error: {final_max_error:.6f}, MSE: {mse:.6f}', fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_filename = os.path.join(layer_save_dir, f'{relu_layer_name.replace(".", "_")}_composite_fit_channels.jpg')
        plt.savefig(save_filename, dpi=200)
        plt.close(fig)
        print(f"  🖼️ 拟合结果图已保存到 {save_filename}")

        return {
            'status': 'success' if result.success else 'failed',
            'message': result.message,
            'relu_layer_name': relu_layer_name,
            'degree_f1': args.degree_f1,
            'degree_f2': args.degree_f2,
            'f1_coeffs': optimized_coeffs_f1.tolist(),
            'f2_coeffs': optimized_coeffs_f2.tolist(),
            'x_min': float(x_min),
            'x_max': float(x_max),
            'final_max_error': float(final_max_error),
            'mse': float(mse),
            'optimization_success': bool(result.success)
        }

    for relu_layer_name in relu_layer_names:
        result = fit_single_relu_layer_updated(args.pth_file_path, relu_layer_name, args, args.save_dir)
        all_relu_fit_results[relu_layer_name] = result

    # 3. 保存所有拟合结果到 JSON
    coeffs_output_path = os.path.join(args.save_dir, 'all_relu_minimax_composite_coeffs.json')
    with open(coeffs_output_path, 'w') as f:
        json.dump(all_relu_fit_results, f, indent=4)
    print(f"\n💾 所有 ReLU 层的拟合系数和信息已保存到 {coeffs_output_path}")

    print("\n--- Minimax 复合拟合脚本执行完毕 ---")