import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import argparse
import os

# IMPORTANT: Directly import resnet20 from your model directory.
# Ensure your 'model' directory is in the Python path or current working directory.
try:
    from resnet import resnet20
except ImportError:
    print("Error: Could not import resnet20 from model/resnet.py.")
    print("Please ensure you have a 'model' directory in the same location as this script,")
    print("and that 'resnet.py' exists within it and defines a 'resnet20' function.")
    exit(1) # Exit if the required model cannot be imported

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 剪枝率计算 (适用于已移除_orig/_mask或未剪枝的模型)
# ==============================================================================
def compute_total_sparsity(model):
    """
    计算并打印模型中所有参数的总稀疏度，以及Conv2d和Linear层权重的稀疏度。
    这个版本适用于已经通过prune.remove()移除_orig/_mask参数的模型，
    或者未经过torch.nn.utils.prune处理的常规模型。
    """
    total_prunable_elements = 0 # Elements in weights of Conv2d/Linear layers
    zero_prunable_elements = 0  # Zeros within those weights
    
    print("\n==> Layer-wise sparsity:")
    for name, module in model.named_modules():
        # Check if the module is a type that can be pruned (Conv2d or Linear)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                # After prune.remove(), module.weight is the final, masked weight.
                # If prune.remove() was NOT called, module.weight is the original weight.
                current_weight = module.weight
                
                current_total_elements = current_weight.numel()
                current_zero_elements = torch.sum(current_weight == 0).item()

                total_prunable_elements += current_total_elements
                zero_prunable_elements += current_zero_elements

                layer_sparsity = (current_zero_elements / current_total_elements) * 100 if current_total_elements > 0 else 0
                
                # Check if it was ever pruned (if _orig still exists, then prune.remove() was not called yet in this function)
                # But in this script, we explicitly call prune.remove() before this function.
                # So we just report it as a standard weight sparsity.
                print(f"  {name}.weight: {layer_sparsity:.2f}% zeroed")
            
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


def main():
    parser = argparse.ArgumentParser(description='Check Model Weight Sparsity after Integrating Pruning')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the fine-tuned model checkpoint (.pth file) containing _orig and _mask parameters')
    parser.add_argument('--save_integrated_model_path', type=str, default=None, help='Optional: Path to save the model after integrating pruning (removing _orig/_mask)')
    
    args = parser.parse_args()

    # 1. Load the model architecture
    model = resnet20().to(device)

    # 2. Apply a dummy pruning to all relevant layers to initialize _orig and _mask attributes.
    # This is crucial for load_state_dict to map correctly.
    print("\nApplying dummy pruning to model layers to prepare for loading pruned state_dict...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Apply a small unstructured pruning to create the _orig and _mask attributes
            prune.l1_unstructured(module, name='weight', amount=0.01)
    print("Dummy pruning applied.")


    # 3. Load the trained model state_dict which contains _orig and _mask parameters
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        exit(1)

    print(f"\nLoading model from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        state_dict = checkpoint['net'] if isinstance(checkpoint, dict) and 'net' in checkpoint else checkpoint

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
        exit(1)

    # 4. Integrate pruning: Remove _orig and _mask parameters, apply masks permanently
    print("\nIntegrating pruning results (removing _orig and _mask parameters)...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
                print(f"  Removed pruning reparametrization for {name}.weight")
    print("Pruning integration complete.")

    model.eval() # Set model to evaluation mode

    # 5. Optional: Save the integrated model
    if args.save_integrated_model_path:
        os.makedirs(os.path.dirname(args.save_integrated_model_path), exist_ok=True)
        torch.save(model.state_dict(), args.save_integrated_model_path)
        print(f"\nIntegrated model saved to: {args.save_integrated_model_path}")

    # 6. Compute and print sparsity on the integrated model
    print("\n--- Computing Model Sparsity on Integrated Model ---")
    compute_total_sparsity(model)
    print("\n--- Sparsity Check Complete ---")

if __name__ == '__main__':
    main()