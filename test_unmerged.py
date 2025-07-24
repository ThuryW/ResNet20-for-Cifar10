import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import argparse

# IMPORTANT: Directly import resnet20 from your model directory.
# Ensure your 'model' directory is in the Python path or current working directory.
try:
    from model.resnet import resnet20
except ImportError:
    print("Error: Could not import resnet20 from model/resnet.py.")
    print("Please ensure you have a 'model' directory in the same location as this script,")
    print("and that 'resnet.py' exists within it and defines a 'resnet20' function.")
    exit(1) # Exit if the required model cannot be imported

# ==============================================================================
# Device Configuration
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AvgMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# ==============================================================================
# Data Loading Function
# ==============================================================================
def load_data(args):
    """Loads CIFAR-10 test dataset."""
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return testloader

# ==============================================================================
# Test Function
# ==============================================================================
def test_model(model, test_loader, description="Test"):
    """Evaluates model performance."""
    model.eval()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    with torch.no_grad():
        for image_batch, gt_batch in tqdm(test_loader, desc=description):
            inputs, targets = image_batch.to(device), gt_batch.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())
            loss_meter.add(loss.item(), inputs.size(0))
            acc = (outputs.argmax(dim=-1).long() == targets).float().mean()
            acc_meter.add(acc.item(), inputs.size(0))

    test_loss = loss_meter.avg
    test_acc = acc_meter.avg

    print(f"--- {description} Result --- Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

# ==============================================================================
# 稀疏度计算 (适用于任何状态的模型)
# ==============================================================================
def compute_total_sparsity(model):
    """
    计算并打印模型中所有参数的总稀疏度，以及Conv2d和Linear层权重的稀疏度。
    此函数能正确处理包含_orig和_mask的模型，也适用于已通过prune.remove()固化的模型。
    """
    total_prunable_elements = 0 # Elements in weights of Conv2d/Linear layers
    zero_prunable_elements = 0  # Zeros within those weights
    
    print("\n==> Layer-wise sparsity:")
    for name, module in model.named_modules():
        # Check if the module is a type that can be pruned (Conv2d or Linear)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                # module.weight 访问的已经是 reparameterized 后的权重（如果存在_orig/_mask），
                # 或者就是原始权重（如果不存在_orig/_mask）。因此直接检查它即可。
                current_weight = module.weight 
                
                current_total_elements = current_weight.numel()
                current_zero_elements = torch.sum(current_weight == 0).item()

                total_prunable_elements += current_total_elements
                zero_prunable_elements += current_zero_elements

                layer_sparsity = (current_zero_elements / current_total_elements) * 100 if current_total_elements > 0 else 0
                
                # Check if it's currently managed by prune API for logging clarity
                status = "Pruned and masked" if prune.is_pruned(module) else "Solidified or never pruned"
                print(f"  {name}.weight: {layer_sparsity:.2f}% zeroed ({status})")
            
            # Handle bias if it exists and is part of the module's parameters
            if hasattr(module, 'bias') and module.bias is not None:
                bias_num_zeros = torch.sum(module.bias == 0).item()
                bias_total_elements = module.bias.numel()
                bias_sparsity = (bias_num_zeros / bias_total_elements) * 100 if bias_total_elements > 0 else 0
                # print(f"  {name}.bias: {bias_sparsity:.2f}% zeroed (Bias)") # Uncomment if you want to see bias sparsity for each layer
        # For other modules like BatchNorm, their parameters are included in the overall model sparsity
        
    if total_prunable_elements > 0:
        overall_prunable_sparsity = zero_prunable_elements / total_prunable_elements
        print(f"\n==> Overall sparsity for PRUNABLE weights (Conv2d & Linear): {overall_prunable_sparsity * 100:.2f}%")
    else:
        print("\nNo prunable weights (Conv2d & Linear) found to calculate specific sparsity, or they have 0 elements.")

    # Calculate total model sparsity including ALL parameters (weights, biases, batchnorm, etc.)
    total_all_params = 0
    zero_all_params = 0
    for param in model.parameters():
        total_all_params += param.numel()
        zero_all_params += torch.sum(param == 0).item()
        
    overall_model_sparsity = zero_all_params / total_all_params if total_all_params > 0 else 0
    print(f"==> Total model sparsity (including ALL parameters): {overall_model_sparsity * 100:.2f}%")
    return overall_model_sparsity


# ==============================================================================
# 主流程
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Test and Solidify Pruned Model')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the model checkpoint (.pth file) that still contains _orig and _mask parameters')
    parser.add_argument('--save_solidified_model_path', type=str, default='./final_solidified_model.pth', 
                        help='Optional: Path to save the model after permanently applying pruning (removing _orig/_mask)')
    
    args = parser.parse_args()

    # 1. Load Data
    test_loader = load_data(args)

    # 2. Initialize ResNet20 model
    model = resnet20().to(device)

    # Ensure the model is unwrapped from DataParallel if it was saved that way
    if isinstance(model, nn.DataParallel):
        print("Unwrapping model from DataParallel...")
        model = model.module

    # 3. Apply dummy pruning to the newly initialized model
    # This is CRUCIAL for loading a state_dict that contains _orig and _mask parameters.
    # It creates the _orig and _mask attributes that the state_dict expects.
    print("\nApplying dummy pruning to model layers to prepare for loading pruned state_dict...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Apply a small unstructured pruning to create the _orig and _mask attributes
            prune.l1_unstructured(module, name='weight', amount=0.01) # Small, non-zero amount to ensure attributes are created
    print("Dummy pruning applied to base model.")

    # 4. Load the trained model state_dict which contains _orig and _mask parameters
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        exit(1)

    print(f"\nLoading unmerged model from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        # Handle cases where state_dict might be directly the checkpoint or under a 'net' key
        state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint

        # Adjust state_dict keys for potential DataParallel prefix (if saved from a DP model)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("Model state_dict loaded successfully.")
    except Exception as e:
        print(f"Error loading model state_dict from {args.model_path}: {e}")
        print("Please ensure the file is a valid PyTorch checkpoint and matches the model architecture.")
        print("Also ensure that the checkpoint indeed contains _orig and _mask parameters if expected.")
        exit(1)

    model.eval() # Set model to evaluation mode

    # --- NEW STEP: Permanently apply pruning to the model (solidify weights) ---
    print("\n--- Permanently applying pruning to the model (prune.remove()) ---")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module): # Only remove if it was actually pruned
                prune.remove(module, 'weight')
                # print(f"  Removed pruning reparametrization for {name}.weight") # Uncomment for detailed log
    print("Pruning permanently applied.")
    # --- END NEW STEP ---

    # Optional: Save the solidified model
    if args.save_solidified_model_path:
        os.makedirs(os.path.dirname(args.save_solidified_model_path), exist_ok=True)
        # Save the state_dict of the model AFTER prune.remove()
        torch.save(model.state_dict(), args.save_solidified_model_path)
        print(f"\nSolidified model saved to: {args.save_solidified_model_path}")


    # Wrap the model in DataParallel for evaluation if using CUDA
    if device == 'cuda':
        print("\nWrapping model in DataParallel for evaluation...")
        model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True # Optimize cuDNN operations

    # 5. Compute and print sparsity on the solidified model
    print("\n--- Computing Model Sparsity on Solidified Model ---")
    compute_total_sparsity(model) 
    
    # 6. Evaluate accuracy of the solidified model
    print("\n--- Evaluating Accuracy of Solidified Model ---")
    test_loss, test_acc = test_model(model, test_loader, description="Solidified Pruned Model")

    print("\n--- Model Test and Solidification Complete ---")

if __name__ == '__main__':
    main()