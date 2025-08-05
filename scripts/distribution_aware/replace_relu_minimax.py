import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import torchvision
import torchvision.transforms as transforms
from resnet import resnet20 # 确保 resnet.py 在你的路径中或已正确导入
import json
from collections import OrderedDict

# --- PyTorch 多项式求值函数 ---
def poly_eval_torch(x, coeffs):
    """
    使用 PyTorch 操作计算多项式值 (Horner 法则)。
    coeffs: torch.Tensor, 降幂排列 [c_n, c_{n-1}, ..., c_0]
    """
    if coeffs.numel() == 0: # Check if tensor is empty
        return torch.zeros_like(x)
    
    # Ensure coeffs are 1D
    coeffs = coeffs.flatten()

    result = coeffs[0] * x
    for i in range(1, coeffs.shape[0] - 1):
        result = result + coeffs[i]
        result = result * x
    result = result + coeffs[-1]
    return result

# --- 自定义复合多项式 ReLU 激活函数模块 ---
class CompositePolynomialReLU(nn.Module):
    def __init__(self, relu_layer_name, f1_coeffs, f2_coeffs, x_min, x_max, original_relu_module=None):
        super(CompositePolynomialReLU, self).__init__()
        self.relu_layer_name = relu_layer_name
        
        # 将 NumPy 数组转换为 torch.Tensor，注册为 buffer
        # buffer 不会被训练，但会随模型移动到 GPU
        self.register_buffer('f1_coeffs', torch.tensor(f1_coeffs, dtype=torch.float32))
        self.register_buffer('f2_coeffs', torch.tensor(f2_coeffs, dtype=torch.float32))
        self.register_buffer('x_min', torch.tensor(x_min, dtype=torch.float32))
        self.register_buffer('x_max', torch.tensor(x_max, dtype=torch.float32))
        
        # 引用原始 ReLU 模块，以防需要回退或调试
        self.original_relu_module = original_relu_module 

    def forward(self, x):
        # 将输入值限制在拟合范围内，防止多项式外推产生的大误差
        x_clipped = torch.clamp(x, self.x_min, self.x_max)

        # 计算 f2(x)
        y_intermediate = poly_eval_torch(x_clipped, self.f2_coeffs)

        # 计算 f1(f2(x))
        out = poly_eval_torch(y_intermediate, self.f1_coeffs)
        
        return out

# --- 替换 ReLU 函数的递归辅助函数 ---
def _replace_relu_recursive_helper(module, poly_coeffs_data, current_name_prefix=""):
    for name, child in module.named_children():
        # 构建当前 ReLU 层的完整路径名
        full_name = f"{current_name_prefix}.{name}" if current_name_prefix else name

        if isinstance(child, nn.ReLU):
            # 检查完整路径名是否在拟合数据中且状态为成功
            if full_name in poly_coeffs_data and poly_coeffs_data[full_name]['status'] == 'success':
                f1_coeffs_np = np.array(poly_coeffs_data[full_name]['f1_coeffs'], dtype=np.float32)
                f2_coeffs_np = np.array(poly_coeffs_data[full_name]['f2_coeffs'], dtype=np.float32)
                
                # 转换为 PyTorch Tensor
                f1_coeffs_tensor = torch.from_numpy(f1_coeffs_np)
                f2_coeffs_tensor = torch.from_numpy(f2_coeffs_np)
                
                x_min = poly_coeffs_data[full_name]['x_min']
                x_max = poly_coeffs_data[full_name]['x_max']
                
                new_relu_module = CompositePolynomialReLU(full_name, f1_coeffs_tensor, f2_coeffs_tensor, x_min, x_max, original_relu_module=child)
                # 使用 setattr 替换模块
                setattr(module, name, new_relu_module)
                print(f"✅ 成功替换层 '{full_name}' 为 CompositePolynomialReLU。")
            else:
                # 打印警告时也使用完整的路径名
                print(f"⚠️ 警告：层 '{full_name}' 的拟合数据缺失或拟合失败，将保留原始 nn.ReLU。")
        elif len(list(child.children())) > 0: # 如果是容器模块，则递归调用
            # 递归调用时，传递新的前缀
            _replace_relu_recursive_helper(child, poly_coeffs_data, full_name)
    return module

# --- 主替换函数 ---
def replace_relu_with_composite_polynomial(model, poly_coeffs_data):
    # 从根模块开始递归替换
    return _replace_relu_recursive_helper(model, poly_coeffs_data)

# --- 测试模型准确率的函数 ---
def test_model_accuracy(model, batch_size, data_dir):
    """
    测试给定模型的准确率。
    """
    model.eval() # 设置为评估模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    return accuracy

# --- 主程序入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace ReLU functions with composite polynomial approximations in ResNet20 and test accuracy.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/base_models/20_ckpt_92.23.pth",
                        help='Path to the .pth original model file.')
    parser.add_argument('--poly_coeffs_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/hook/minimax_composite_relu_all_layers/all_relu_minimax_composite_coeffs.json",
                        help='Path to the JSON file containing all ReLU composite polynomial coefficients.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Input batch size for data loading during accuracy test.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--output_model_path', type=str, default=None,
                        help='Optional: Path to save the modified model state_dict.')


    args = parser.parse_args()

    # 1. 加载拟合好的多项式系数
    if not os.path.exists(args.poly_coeffs_path):
        print(f"❌ 错误：未找到多项式系数文件：{args.poly_coeffs_path}")
        exit()
    
    with open(args.poly_coeffs_path, 'r') as f:
        poly_coeffs_data = json.load(f)
    print(f"✨ 成功加载多项式系数文件：{args.poly_coeffs_path}")

    # 2. 加载原始模型和权重
    original_model = resnet20()
    try:
        checkpoint = torch.load(args.pth_file_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        original_model.load_state_dict(new_state_dict)
        print(f"✨ 成功加载原始模型权重：{args.pth_file_path}")
    except Exception as e:
        print(f"❌ 加载原始模型权重失败：{e}")
        print("请确保你的 .pth 文件与 ResNet20 模型结构匹配。")
        exit()

    # 3. 替换模型中的 ReLU 函数
    # 注意：这里我们对原始模型的副本进行替换，以保留原始模型
    modified_model = replace_relu_with_composite_polynomial(original_model, poly_coeffs_data)
    
    # 4. 测试修改后模型的准确率
    print("\n🚀 开始测试替换 ReLU 后的模型准确率...")
    accuracy = test_model_accuracy(modified_model, args.batch_size, args.data_dir)
    print(f"\n✅ 替换 ReLU 为复合多项式后，模型在测试集上的准确率: {accuracy:.2f}%")

    # 5. 可选：保存修改后的模型权重
    if args.output_model_path:
        os.makedirs(os.path.dirname(args.output_model_path) or '.', exist_ok=True)
        torch.save(modified_model.state_dict(), args.output_model_path)
        print(f"💾 修改后的模型权重已保存到：{args.output_model_path}")

    print("\n--- 替换 ReLU 脚本执行完毕 ---")