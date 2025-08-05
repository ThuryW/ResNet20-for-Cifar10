import torch
import torch.nn as nn
import numpy as np
import os
import argparse
import torchvision
import torchvision.transforms as transforms
from resnet import resnet20 # Make sure resnet.py is in your path or correctly imported
import json
from collections import OrderedDict
from tqdm import tqdm # 导入 tqdm

# Define the custom PolynomialReLU activation function module
class PolynomialReLU(nn.Module):
    def __init__(self, relu_layer_name, poly_coeffs, x_min, x_max, original_relu_module=None):
        super(PolynomialReLU, self).__init__()
        self.relu_layer_name = relu_layer_name
        # Store coefficients as a torch.Tensor for computation on device
        # np.poly1d expects coeffs in descending power, so direct use
        self.coeffs = torch.tensor(poly_coeffs, dtype=torch.float32)
        self.x_min = x_min
        self.x_max = x_max
        # Keep a reference to the original ReLU if needed for fallback
        self.original_relu_module = original_relu_module # For cases where fitting failed

    def forward(self, x):
        # Move coefficients to the same device as input x
        coeffs_on_device = self.coeffs.to(x.device)

        # Clip input values to the range observed during fitting
        # This prevents extrapolation issues with polynomials
        x_clipped = torch.clamp(x, self.x_min, self.x_max)

        # Compute polynomial value: p[0]*x**(N-1) + ... + p[N-1]
        # Equivalent to numpy.polyval(coeffs, x_clipped) but in PyTorch
        # Iterating through coefficients for polynomial evaluation
        
        # Handle empty coeffs (fit failed cases) or fallback to original ReLU
        if self.coeffs.numel() == 0: # If no coefficients were saved (fit failed)
            if self.original_relu_module:
                return self.original_relu_module(x) # Fallback to original ReLU
            else:
                return torch.relu(x) # Fallback to torch.relu
                
        # Calculate polynomial using Horner's method: C_n * x^n + C_{n-1} * x^{n-1} + ... + C_0
        # poly_coeffs from np.polyfit are [c_n, c_{n-1}, ..., c_0] (highest degree first)
        # So, coeffs_on_device[0] is for x^(N-1), coeffs_on_device[1] for x^(N-2), etc.
        
        output = coeffs_on_device[0] * x_clipped
        # Iterate from the second highest degree (index 1) up to the second last coefficient
        for i in range(1, len(coeffs_on_device) - 1):
            output = output + coeffs_on_device[i] # Add the current coefficient
            output = output * x_clipped         # Multiply by x for the next term's power
        
        # Finally, add the constant term (the last coefficient)
        output = output + coeffs_on_device[-1] 

        return output

# Function to replace nn.ReLU modules with PolynomialReLU
def replace_relu_with_polynomial(model, poly_coeffs_data):
    """
    Recursively replaces nn.ReLU modules in the model with PolynomialReLU instances.
    poly_coeffs_data: Dictionary loaded from relu_poly_coeffs.json
                      Key: ReLU layer name (e.g., 'relu_initial', 'layer1.0.relu')
                      Value: {'coeffs': [], 'min_val': float, 'max_val': float, ...}
    """
    replaced_count = 0
    # Create a list to store (parent_module, attribute_name, original_relu_module)
    # We collect them first to avoid modifying the dictionary while iterating
    modules_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # Check if this ReLU layer has corresponding fitted coefficients
            if name in poly_coeffs_data and poly_coeffs_data[name]["status"] == "success":
                coeffs_info = poly_coeffs_data[name]
                
                # Retrieve parent module and attribute name for replacement
                parts = name.rsplit('.', 1)
                if len(parts) == 1: # Top-level module (e.g., 'relu_initial')
                    parent_module = model
                    attr_name = parts[0]
                else: # Nested module (e.g., 'layer1.0.relu1')
                    parent_path = parts[0]
                    attr_name = parts[1]
                    parent_module = model
                    for p in parent_path.split('.'):
                        parent_module = getattr(parent_module, p)
                
                # Append to list for later replacement
                modules_to_replace.append((parent_module, attr_name, module, coeffs_info))
            else:
                print(f"⚠️ Warning: ReLU '{name}' did not have successful polynomial fit. Keeping original ReLU.")
                # We could also replace it with a PolynomialReLU that defaults to torch.relu
                # but for simplicity, we keep original nn.ReLU here.

    for parent_module, attr_name, original_relu_module, coeffs_info in modules_to_replace:
        poly_relu_instance = PolynomialReLU(
            relu_layer_name=attr_name, # or full name for more clarity in debug
            poly_coeffs=coeffs_info["coeffs"],
            x_min=coeffs_info["min_val"],
            x_max=coeffs_info["max_val"],
            original_relu_module=original_relu_module # Pass original module for potential fallback
        )
        setattr(parent_module, attr_name, poly_relu_instance)
        replaced_count += 1
        print(f"✅ Replaced ReLU: '{parent_module.__class__.__name__}.{attr_name}' with PolynomialReLU (Layer name: {attr_name})")

    print(f"Total {replaced_count} nn.ReLU modules replaced with PolynomialReLU.")
    return model

# Test model accuracy
def test_model_accuracy(model, batch_size, data_dir):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # ✨ 加入 tqdm 进度条 ✨
        for inputs, labels in tqdm(testloader, desc="Testing Accuracy"):
            # If using GPU, move tensors to GPU
            # inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"\nModel accuracy on test set: {accuracy:.2f}%") # 添加换行符，让进度条结束后输出更清晰
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace ReLU functions with polynomial approximations and test model accuracy.')
    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/base_models/20_ckpt_92.23.pth",
                        help='Path to the .pth model file.')
    parser.add_argument('--poly_coeffs_path', type=str,
                        default="/home/wangtianyu/relu_finetune/hook/all_relu_fits_input/relu_poly_coeffs.json",
                        help='Path to the JSON file containing polynomial coefficients.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for data loading.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation (e.g., "cuda" or "cpu").') # 新增 device 参数
    parser.add_argument('--save_model_path', type=str, default=None,
                        help='Optional: Path to save the modified model (e.g., "modified_resnet20.pth").') # 新增 save_model_path 参数

    args = parser.parse_args()

    # 确定设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # 1. Load polynomial coefficients
    if not os.path.exists(args.poly_coeffs_path):
        print(f"❌ Error: Polynomial coefficients file not found at '{args.poly_coeffs_path}'. Please run fit_relu_LS.py first.")
        exit()
    with open(args.poly_coeffs_path, 'r') as f:
        poly_coeffs_data = json.load(f)
    print(f"✨ Successfully loaded polynomial coefficients from: {args.poly_coeffs_path}")

    # 2. Load original model and weights
    model = resnet20()
    try:
        checkpoint = torch.load(args.pth_file_path, map_location=device) # 确保加载到正确设备
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"✨ Successfully loaded model weights from: {args.pth_file_path}")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        print("Please ensure your .pth file matches the ResNet20 model structure.")
        exit()
    model.to(device) # 将模型移动到指定设备

    # 3. Replace ReLU functions in the model
    print("\n🚀 Starting ReLU replacement...")
    model_modified = replace_relu_with_polynomial(model, poly_coeffs_data)
    
    # 4. Test accuracy of the modified model
    print("\n⚡ Starting accuracy test for the model with PolynomialReLU replacements...")
    accuracy = test_model_accuracy(model_modified, args.batch_size, args.data_dir)
    print(f"🎉 Model with PolynomialReLU tested. Accuracy: {accuracy:.2f}%")

    # 5. 可选：保存修改后的模型
    if args.save_model_path:
        try:
            torch.save(model_modified.state_dict(), args.save_model_path)
            print(f"✅ Modified model saved to: {args.save_model_path}")
        except Exception as e:
            print(f"❌ Error saving modified model: {e}")

    print("\n--- Replacement and Testing Complete ---")