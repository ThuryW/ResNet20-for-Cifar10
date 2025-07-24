import torch
import torch.nn as nn
import argparse
import os

from resnet import resnet20

def calculate_sparsity(model_path):
    """
    Loads a PyTorch model from a checkpoint and calculates its weight sparsity.
    
    Args:
        model_path (str): Path to the .pth checkpoint file.
    
    Returns:
        None: Prints the sparsity details.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        return

    print(f"--- Loading model from: {model_path} ---")
    
    # 1. Instantiate the model architecture
    # We need an instance of the model to load the state_dict into.
    # Make sure this matches the model you saved (e.g., resnet20)
    model = resnet20() 
    
    # 2. Load the state_dict from the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location='cpu') # Load to CPU to avoid GPU memory issues if not needed
        
        # Check if the state_dict is nested under a 'net' key or is the dict itself
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        else:
            state_dict = checkpoint # Assume the checkpoint is just the state_dict
            
        # Handle 'module.' prefix if the model was saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict)
        print("Model state_dict loaded successfully.")
        
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Please ensure the model architecture (e.g., resnet20) matches the saved checkpoint.")
        return

    print("\n--- Calculating Sparsity ---")
    
    total_prunable_elements = 0
    zero_prunable_elements = 0
    
    # Iterate through all modules (layers) in the model
    for name, module in model.named_modules():
        # We are interested in Conv2d and Linear layers for weight sparsity
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight'): # Check if the layer has a 'weight' attribute
                weight_tensor = module.weight.data # Get the actual weight tensor
                
                layer_total_elements = weight_tensor.nelement() # Total elements in this layer's weight
                layer_zero_elements = torch.sum(weight_tensor == 0).item() # Count zeros in this layer
                
                layer_sparsity = (layer_zero_elements / layer_total_elements) * 100 if layer_total_elements > 0 else 0
                
                print(f"  Layer: {name} | Type: {type(module).__name__} | "
                      f"Shape: {list(weight_tensor.shape)} | "
                      f"Sparsity: {layer_sparsity:.2f}% ({layer_zero_elements}/{layer_total_elements})")
                
                total_prunable_elements += layer_total_elements
                zero_prunable_elements += layer_zero_elements
                
            else:
                print(f"  Layer: {name} ({type(module).__name__}) has no 'weight' attribute to check.")

    # Calculate overall sparsity
    overall_sparsity = (zero_prunable_elements / total_prunable_elements) * 100 if total_prunable_elements > 0 else 0
    print(f"\n--- Overall Prunable Layer Sparsity: {overall_sparsity:.2f}% ({zero_prunable_elements}/{total_prunable_elements}) ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate weight sparsity of a PyTorch model checkpoint.')
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the .pth checkpoint file of the pruned model.')
    
    args = parser.parse_args()
    
    calculate_sparsity(args.model_path)