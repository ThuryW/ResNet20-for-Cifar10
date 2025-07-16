import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import torchvision
import torchvision.transforms as transforms

# Assuming your resnet20 model definition is in model/resnet.py
from model.resnet import resnet20 

# Define AvgMeter utility class if not already in utils.py
class AvgMeter(object):
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


# Configure device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# Data Loading Function
# ==============================================================================
def load_data(args):
    """
    Loads CIFAR-10 training and test datasets.
    """
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# ==============================================================================
# FINAL CORRECTED Custom Grouped Unstructured Pruning Function
# ==============================================================================
def grouped_unstructured_prune(module, name, amount, group_size):
    """
    Applies unstructured pruning such that pruning patterns are identical
    across groups of output channels.
    Assumes `module.weight` is of shape (out_channels, in_channels, kernel_h, kernel_w).
    
    This function directly calculates the mask for a representative kernel within each group,
    then replicates that *exact same mask* to all other channels/kernels in that group.
    """
    weight_tensor = getattr(module, name) # Get the original weight parameter
    out_channels, in_channels, k_h, k_w = weight_tensor.shape

    if out_channels % group_size != 0:
        print(f"Warning: Output channels ({out_channels}) of {name} is not a multiple of group_size ({group_size}). Applying standard unstructured pruning for this module.")
        prune.l1_unstructured(module, name=name, amount=amount)
        return

    num_groups = out_channels // group_size
    final_mask = torch.ones_like(weight_tensor, dtype=torch.bool) # Initialize with True (no pruning), boolean mask

    # Iterate through each group
    for i in range(num_groups):
        start_channel = i * group_size
        
        # Extract the representative *kernel* for this group (e.g., the first channel's kernel within the group)
        # This is a tensor of shape (in_channels, k_h, k_w)
        # We work on the absolute values to find the smallest elements.
        representative_kernel_values = weight_tensor[start_channel].abs().flatten()
        
        # Determine the threshold for pruning.
        # Find the value at the (amount * num_elements)th position when sorted.
        num_elements_in_kernel = representative_kernel_values.numel()
        num_prune_in_kernel = int(amount * num_elements_in_kernel)
        
        if num_prune_in_kernel == 0: # Avoid issues if amount is too small
            kernel_mask = torch.ones_like(weight_tensor[start_channel], dtype=torch.bool)
        else:
            # Sort and find the threshold for the smallest 'amount' elements
            sorted_values, _ = torch.sort(representative_kernel_values)
            threshold = sorted_values[num_prune_in_kernel - 1] # Get the value at the cutoff point

            # Create the mask: True if element should be kept, False if pruned
            kernel_mask = (weight_tensor[start_channel].abs() >= threshold).to(torch.bool)
        
        # Apply this kernel_mask to all channels within the current group in the final_mask
        for j in range(group_size):
            final_mask[start_channel + j] = kernel_mask
            
    # Apply the custom mask using prune.CustomSparsity
    # PyTorch's prune functions expect a float mask where 0 means prune.
    # So convert boolean mask (True=keep, False=prune) to float mask (1.0=keep, 0.0=prune)
    prune.custom_from_mask(module, name=name, mask=final_mask.float().to(weight_tensor.device))


# ==============================================================================
# Pruning Function with better naming checks and logging
# ==============================================================================
def prune_model_with_strategy(model, args):
    """
    Applies unstructured and semi-structured pruning based on specified strategies.
    """
    print(f"\n--- Applying Pruning Strategy ---")

    # Define pruning amounts for different layers
    prune_rates = {
        'init_layer': args.prune_rate_init_layer, # >= 80%
        'layer1': args.prune_rate_layer1,         # >= 80%
        'layer2': args.prune_rate_layer2,         # e.g., 50%
        'layer3': args.prune_rate_layer3          # e.g., 30%
    }
    
    # Iterate through model modules and apply pruning
    pruned_modules_count = 0
    for name, module in model.named_modules():
        # IMPORTANT: `name` will include "module." prefix if DataParallel is used.
        # We need to handle this for matching.
        clean_name = name.replace('module.', '') 

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            print(f"Processing module: {name} (Cleaned: {clean_name})")

            # Init Layer (conv1) - Unstructured
            if clean_name == 'conv1':
                print(f"  -> Applying unstructured pruning to {name} with {prune_rates['init_layer'] * 100:.2f}%")
                prune.l1_unstructured(module, name='weight', amount=prune_rates['init_layer'])
                pruned_modules_count += 1
            
            # Layer1 (e.g., layer1.0.conv1, layer1.0.conv2, etc.) - Unstructured
            elif clean_name.startswith('layer1.'):
                print(f"  -> Applying unstructured pruning to {name} with {prune_rates['layer1'] * 100:.2f}%")
                prune.l1_unstructured(module, name='weight', amount=prune_rates['layer1'])
                pruned_modules_count += 1
            
            # Layer2 (e.g., layer2.0.conv1, layer2.0.conv2, etc.) - Semi-structured (4-channel grouped)
            elif clean_name.startswith('layer2.'):
                print(f"  -> Applying 4-channel grouped unstructured pruning to {name} with {prune_rates['layer2'] * 100:.2f}%")
                grouped_unstructured_prune(module, name='weight', amount=prune_rates['layer2'], group_size=4)
                pruned_modules_count += 1
            
            # Layer3 (e.g., layer3.0.conv1, layer3.0.conv2, etc.) - Semi-structured (16-channel grouped)
            elif clean_name.startswith('layer3.'):
                print(f"  -> Applying 16-channel grouped unstructured pruning to {name} with {prune_rates['layer3'] * 100:.2f}%")
                grouped_unstructured_prune(module, name='weight', amount=prune_rates['layer3'], group_size=16)
                pruned_modules_count += 1
            
            # For the final linear layer (if any)
            elif clean_name == 'linear': 
                print(f"  -> Applying unstructured pruning to {name} with {prune_rates['init_layer'] * 100:.2f}% (using init_layer rate)")
                prune.l1_unstructured(module, name='weight', amount=prune_rates['init_layer'])
                pruned_modules_count += 1
            
            else:
                print(f"  -> Skipping pruning for {name} (no specific rule applied)")
        else:
            pass # Skip non-prunable modules

    print(f"\n--- Pruning application complete. {pruned_modules_count} modules were targeted for pruning. ---")

    # Calculate and print sparsity for each *pruned* module AFTER application
    # This loop uses the actual weight_mask created by prune.
    print("\n--- Detailed Sparsity After Pruning Application ---")
    total_pruned_elements = 0
    zero_pruned_elements = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if prune.is_pruned(module): # Check if this module actually has pruning reparameterization
                # The mask directly tells us the sparsity
                mask = getattr(module, 'weight_mask')
                layer_total_elements = mask.nelement()
                layer_zero_elements = torch.sum(mask == 0).item()
                layer_sparsity = (layer_zero_elements / layer_total_elements) * 100 if layer_total_elements > 0 else 0
                print(f"  Module: {name} | Mask Sparsity: {layer_sparsity:.2f}% ({layer_zero_elements}/{layer_total_elements})")
                
                total_pruned_elements += layer_total_elements
                zero_pruned_elements += layer_zero_elements
            # else:
            #     print(f"  Module: {name} is NOT marked as pruned by PyTorch.")


    actual_overall_sparsity = (zero_pruned_elements / total_pruned_elements) * 100 if total_pruned_elements > 0 else 0
    print(f"\nOverall Model Sparsity (of PRUNED layers, based on mask): {actual_overall_sparsity:.2f}%")

    return model

# ==============================================================================
# Fine-tuning Function (No change)
# ==============================================================================
def finetune_model(model, train_loader, args):
    """
    Fine-tunes the pruned model.
    """
    print(f"\n--- Starting fine-tuning for {args.finetune_epochs} epochs ---")
    model.train() # Set to training mode
    
    optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.finetune_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.finetune_epochs):
        print(f'Finetune Epoch: {epoch+1}/{args.finetune_epochs}')
        loss_meter = AvgMeter()
        acc_meter = AvgMeter()
        
        with tqdm(total=len(train_loader), desc=f"Finetune Epoch {epoch+1}") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.long())
                loss.backward()
                optimizer.step()

                loss_meter.add(loss.item(), inputs.size(0))
                acc = (outputs.argmax(dim=-1).long() == targets).float().mean()
                acc_meter.add(acc.item(), inputs.size(0))

                pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg * 100})
                pbar.update(1)

        scheduler.step()
        print(f"Finetune Epoch {epoch+1} finished. Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg * 100:.2f}%")
    
    print("--- Fine-tuning complete ---")
    return model

# ==============================================================================
# Test Function (Corrected)
# ==============================================================================
def test_model(model, test_loader, description="Test"):
    """
    Evaluates model performance.
    """
    model.eval() # Set to evaluation mode
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    with torch.no_grad():
        for image_batch, gt_batch in tqdm(test_loader, desc=description):
            image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
            gt_batch = gt_batch.long()
            pred_batch = model(image_batch)
            loss = criterion(pred_batch, gt_batch)
            loss_meter.add(loss.item(), image_batch.size(0))
            acc = (pred_batch.argmax(dim=-1).long() == gt_batch).float().mean()
            acc_meter.add(acc.item(), image_batch.size(0))
    test_loss = loss_meter.avg # Removed ()
    test_acc = acc_meter.avg   # Removed ()

    print(f"--- {description} Result --- Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

# ==============================================================================
# Save Model Function (with final_test_acc parameter fix)
# ==============================================================================
def save_model(model, args, final_test_acc, suffix="pruned_finetuned"):
    """
    Saves the fine-tuned model after removing pruning reparametrizations.
    """
    save_dir = 'pruned_checkpoints'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Remove pruning reparametrizations and make the sparsity permanent
    # Iterate through named_modules to handle DataParallel cases
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and prune.is_pruned(module):
            # This 'prune.remove' will make the zeros permanent in module.weight
            prune.remove(module, 'weight') 

    # If model is DataParallel wrapped, save its internal .module
    if isinstance(model, nn.DataParallel):
        state = {
            'net': model.module.state_dict(),
            'pruning_rates': {
                'init_layer': args.prune_rate_init_layer,
                'layer1': args.prune_rate_layer1,
                'layer2': args.prune_rate_layer2,
                'layer3': args.prune_rate_layer3
            },
            'finetune_epochs': args.finetune_epochs,
            'final_test_acc': final_test_acc 
        }
    else:
        state = {
            'net': model.state_dict(),
            'pruning_rates': {
                'init_layer': args.prune_rate_init_layer,
                'layer1': args.prune_rate_layer1,
                'layer2': args.prune_rate_layer2,
                'layer3': args.prune_rate_layer3
            },
            'finetune_epochs': args.finetune_epochs,
            'final_test_acc': final_test_acc 
        }
    
    # Construct filename based on pruning amounts
    filename = (f"resnet20_{suffix}"
                f"_init{int(args.prune_rate_init_layer*100)}"
                f"_l1{int(args.prune_rate_layer1*100)}"
                f"_l2{int(args.prune_rate_layer2*100)}"
                f"_l3{int(args.prune_rate_layer3*100)}"
                f"_ft{args.finetune_epochs}.pth")
    filepath = os.path.join(save_dir, filename)
    
    torch.save(state, filepath)
    print(f"\nModel saved to: {filepath}")

# ==============================================================================
# Main Function
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning and Fine-tuning with Custom Strategies')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for data loaders')
    parser.add_argument('--model_path', default='./checkpoint/20_ckpt_92.23.pth', type=str, help='path to original trained model checkpoint')
    
    # Pruning rates for different layers
    parser.add_argument('--prune_rate_init_layer', default=0.9, type=float, help='Pruning rate for init layer (conv1)')
    parser.add_argument('--prune_rate_layer1', default=0.9, type=float, help='Pruning rate for layer1 (blocks)')
    parser.add_argument('--prune_rate_layer2', default=0.8, type=float, help='Pruning rate for layer2 (blocks) with 4-channel grouping')
    parser.add_argument('--prune_rate_layer3', default=0.5, type=float, help='Pruning rate for layer3 (blocks) with 16-channel grouping')
    
    parser.add_argument('--finetune_epochs', default=20, type=int, help='Number of epochs for fine-tuning after pruning')
    parser.add_argument('--finetune_lr', default=0.01, type=float, help='Learning rate for fine-tuning')
    args = parser.parse_args()

    # 1. Load Data
    train_loader, test_loader = load_data(args)

    # 2. Load Pre-trained Model
    print(f"\n--- Loading pre-trained model from {args.model_path} ---")
    net = resnet20()
    net = net.to(device)

    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['net']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 3. Evaluate Original Model Performance
    print("\n--- Evaluating Original Model ---")
    original_loss, original_acc = test_model(net, test_loader, "Original Model Test")

    # 4. Apply Pruning
    # Note: prune_model_with_strategy will modify the 'net' object in place.
    pruned_net = prune_model_with_strategy(net, args) 

    # 5. Evaluate Pruned (before finetune) Model Performance
    print("\n--- Evaluating Pruned (before finetune) Model ---")
    pruned_loss, pruned_acc = test_model(pruned_net, test_loader, "Pruned Model Test")

    # 6. Fine-tune the Pruned Model
    finetuned_net = finetune_model(pruned_net, train_loader, args)

    # 7. Evaluate Fine-tuned Model Performance
    print("\n--- Evaluating Fine-tuned Model ---")
    final_loss, final_acc = test_model(finetuned_net, test_loader, "Fine-tuned Model Test")

    # 8. Save the Fine-tuned Model
    save_model(finetuned_net, args, final_acc, suffix="pruned_finetuned")

    print("\n--- Pruning and Fine-tuning Process Complete ---")
    print(f"Original Accuracy: {original_acc * 100:.2f}%")
    print(f"Pruned (no finetune) Accuracy: {pruned_acc * 100:.2f}%")
    print(f"Fine-tuned Accuracy: {final_acc * 100:.2f}%")