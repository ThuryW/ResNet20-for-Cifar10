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

from model.resnet import resnet20 # Make sure resnet20 is defined here

# Define AvgMeter utility class directly here for consistency
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
# CORRECTED Custom Grouped Unstructured Pruning Function
# ==============================================================================
def grouped_unstructured_prune(module, name, amount, group_size):
    """
    Applies unstructured pruning such that pruning patterns are identical
    across groups of output channels.
    Assumes `module.weight` is of shape (out_channels, in_channels, kernel_h, kernel_w).
    """
    weight = getattr(module, name)
    out_channels, in_channels, k_h, k_w = weight.shape

    if out_channels % group_size != 0:
        print(f"Warning: Output channels ({out_channels}) of {name} is not a multiple of group_size ({group_size}). Skipping grouped pruning for this module and applying standard unstructured pruning.")
        prune.l1_unstructured(module, name=name, amount=amount)
        return

    num_groups = out_channels // group_size
    final_mask = torch.ones_like(weight) 

    for i in range(num_groups):
        start_channel = i * group_size
        
        representative_kernel = weight.data[start_channel]
        dummy_param = torch.nn.Parameter(representative_kernel.clone())
        dummy_module = nn.Module()
        dummy_module.dummy_weight = dummy_param
        
        prune.l1_unstructured(dummy_module, name='dummy_weight', amount=amount)
        
        kernel_mask = dummy_module.dummy_weight_mask
        
        for j in range(group_size):
            final_mask[start_channel + j] = kernel_mask
            
    prune.custom_from_mask(module, name=name, mask=final_mask.to(weight.device))


# ==============================================================================
# Pruning Function (Modified to correctly apply both pruning types)
# ==============================================================================
def prune_model_with_strategy(model, args):
    """
    Applies unstructured, semi-structured, and now channel-wise structured pruning
    based on specified strategies.
    """
    print(f"\n--- Applying Pruning Strategy ---")

    # Define unstructured/grouped pruning rates (existing)
    prune_rates = {
        'init_layer': args.prune_rate_init_layer,
        'layer1': args.prune_rate_layer1,
        'layer2': args.prune_rate_layer2,
        'layer3': args.prune_rate_layer3
    }
    
    # Define structured pruning rates (NEW)
    structured_prune_rates = {
        'init_layer': args.prune_rate_structured_init_layer, # For conv1 if specified separately
        'stage1_channels': args.prune_rate_structured_stage1, # For 16-channel layers (conv1, layer1.x.convX)
        'stage2_channels': args.prune_rate_structured_stage2, # For 32-channel layers (layer2.x.convX)
        'stage3_channels': args.prune_rate_structured_stage3, # For 64-channel layers (layer3.x.convX)
    }

    for name, module in model.named_modules():
        # --- FIX: Remove 'module.' prefix for consistent naming comparison ---
        actual_name = name.replace('module.', '')
        
        if isinstance(module, nn.Conv2d):
            current_structured_prune_amount = 0.0 # Default to no structured pruning

            # Determine the structured pruning rate based on the stage/output channels
            if hasattr(module, 'weight'): # Ensure the module has a weight tensor
                out_channels = module.weight.shape[0]

                # Apply structured pruning based on stage/output channels
                if actual_name == 'conv1' and structured_prune_rates['init_layer'] > 0:
                    current_structured_prune_amount = structured_prune_rates['init_layer']
                elif out_channels == 16 and actual_name.startswith('layer1.'):
                    current_structured_prune_amount = structured_prune_rates['stage1_channels']
                elif out_channels == 32 and actual_name.startswith('layer2.'):
                    current_structured_prune_amount = structured_prune_rates['stage2_channels']
                elif out_channels == 64 and actual_name.startswith('layer3.'):
                    current_structured_prune_amount = structured_prune_rates['stage3_channels']
                # Note: Linear layer (module.linear) is not a Conv2d, so this block won't apply.
                # Structured pruning on dim=0 for Linear layer would be pruning output features.

            if current_structured_prune_amount > 0:
                target_channels_to_prune = int(current_structured_prune_amount * out_channels)
                # Ensure at least 1 channel if amount > 0 and out_channels > 0, unless amount leads to 0.
                # If target_channels_to_prune is 0 but current_structured_prune_amount > 0 (e.g. 0.01 * 16 = 0.16 -> 0), don't prune.
                if target_channels_to_prune == 0 and current_structured_prune_amount > 0 and out_channels > 0:
                    print(f"  -> Skipping structured pruning for {name} ({out_channels} channels) as amount {current_structured_prune_amount*100:.2f}% rounds to 0 channels.")
                elif out_channels == 0: # Avoid division by zero if somehow out_channels is 0
                    print(f"  -> Skipping structured pruning for {name} as it has 0 output channels.")
                else:
                    print(f"  -> Applying L1-structured (channel-wise) pruning to {name} ({out_channels} channels) with {current_structured_prune_amount * 100:.2f}% (targeting {target_channels_to_prune} channels)")
                    prune.ln_structured(module, name='weight', amount=current_structured_prune_amount, n=1, dim=0)
            elif current_structured_prune_amount == 0 and actual_name.startswith('layer') or actual_name == 'conv1':
                 print(f"  -> No L1-structured (channel-wise) pruning applied to {name} (rate is 0%).")


        # Now, apply the existing unstructured or grouped unstructured pruning
        # This part ensures that if a module is a Linear or Conv2d, it's processed for these types of pruning.
        if isinstance(module, (nn.Linear, nn.Conv2d)): 
            print(f"Processing module for additional pruning: {name}")

            # Init Layer (conv1) - Unstructured
            if actual_name == 'conv1': # Use actual_name for comparison
                print(f"  -> Applying unstructured pruning to {name} with {prune_rates['init_layer'] * 100:.2f}%")
                prune.l1_unstructured(module, name='weight', amount=prune_rates['init_layer'])
            
            # Layer1 (e.g., layer1.0.conv1, layer1.0.conv2, etc.) - Unstructured
            elif actual_name.startswith('layer1.'): # Use actual_name for comparison
                print(f"  -> Applying unstructured pruning to {name} with {prune_rates['layer1'] * 100:.2f}%")
                prune.l1_unstructured(module, name='weight', amount=prune_rates['layer1'])
            
            # Layer2 (e.g., layer2.0.conv1, layer2.0.conv2, etc.) - Semi-structured (4-channel grouped)
            elif actual_name.startswith('layer2.'): # Use actual_name for comparison
                print(f"  -> Applying 4-channel grouped unstructured pruning to {name} with {prune_rates['layer2'] * 100:.2f}%")
                grouped_unstructured_prune(module, name='weight', amount=prune_rates['layer2'], group_size=4)
            
            # Layer3 (e.g., layer3.0.conv1, layer3.0.conv2, etc.) - Semi-structured (16-channel grouped)
            elif actual_name.startswith('layer3.'): # Use actual_name for comparison
                print(f"  -> Applying 16-channel grouped unstructured pruning to {name} with {prune_rates['layer3'] * 100:.2f}%")
                grouped_unstructured_prune(module, name='weight', amount=prune_rates['layer3'], group_size=16)
            
            elif actual_name == 'linear': # Use actual_name for comparison
                print(f"  -> Applying unstructured pruning to {name} with {prune_rates['init_layer'] * 100:.2f}% (using init_layer rate)")
                prune.l1_unstructured(module, name='weight', amount=prune_rates['init_layer'])
            
            else:
                if not isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AvgPool2d, nn.Dropout)):
                    print(f"  -> Skipping additional pruning for {name}")

    total_elements = 0
    zero_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight_mask'):
                total_elements += module.weight_mask.nelement()
                zero_elements += torch.sum(module.weight_mask == 0)
            elif hasattr(module, 'weight'): # Fallback if no mask (e.g., if no pruning applied)
                total_elements += module.weight.nelement()
                zero_elements += torch.sum(module.weight == 0)

    actual_sparsity = (zero_elements / total_elements) * 100 if total_elements > 0 else 0
    print(f"\nOverall Model sparsity after all pruning: {actual_sparsity:.2f}%")

    return model

# ==============================================================================
# Fine-tuning Function (Fixed AvgMeter calls)
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
# Test Function (Fixed AvgMeter calls)
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
            
    test_loss = loss_meter.avg
    test_acc = acc_meter.avg

    print(f"--- {description} Result --- Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

# ==============================================================================
# Save Model Function
# ==============================================================================
def save_model(model, args, final_test_acc, suffix="pruned_finetuned"):
    """
    Saves the fine-tuned model after removing pruning reparametrizations.
    """
    save_dir = 'pruned_checkpoints'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and prune.is_pruned(module):
            prune.remove(module, 'weight') 

    if isinstance(model, nn.DataParallel):
        state = {
            'net': model.module.state_dict(),
            'pruning_rates': {
                'unstructured_init_layer': args.prune_rate_init_layer,
                'unstructured_layer1': args.prune_rate_layer1,
                'grouped_layer2': args.prune_rate_layer2,
                'grouped_layer3': args.prune_rate_layer3,
                # New structured pruning rates
                'structured_init_layer': args.prune_rate_structured_init_layer,
                'structured_stage1': args.prune_rate_structured_stage1,
                'structured_stage2': args.prune_rate_structured_stage2,
                'structured_stage3': args.prune_rate_structured_stage3,
            },
            'finetune_epochs': args.finetune_epochs,
            'final_test_acc': final_test_acc
        }
    else:
        state = {
            'net': model.state_dict(),
            'pruning_rates': {
                'unstructured_init_layer': args.prune_rate_init_layer,
                'unstructured_layer1': args.prune_rate_layer1,
                'grouped_layer2': args.prune_rate_layer2,
                'grouped_layer3': args.prune_rate_layer3,
                # New structured pruning rates
                'structured_init_layer': args.prune_rate_structured_init_layer,
                'structured_stage1': args.prune_rate_structured_stage1,
                'structured_stage2': args.prune_rate_structured_stage2,
                'structured_stage3': args.prune_rate_structured_stage3,
            },
            'finetune_epochs': args.finetune_epochs,
            'final_test_acc': final_test_acc
        }
    
    # Update filename to reflect new structured pruning rates
    filename = (f"resnet20_{suffix}"
                f"_uni{int(args.prune_rate_init_layer*100)}"
                f"_unl1{int(args.prune_rate_layer1*100)}"
                f"_gl2{int(args.prune_rate_layer2*100)}"
                f"_gl3{int(args.prune_rate_layer3*100)}"
                f"_strI{int(args.prune_rate_structured_init_layer*100)}"
                f"_strS1{int(args.prune_rate_structured_stage1*100)}"
                f"_strS2{int(args.prune_rate_structured_stage2*100)}"
                f"_strS3{int(args.prune_rate_structured_stage3*100)}"
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
    
    # Existing unstructured/grouped unstructured pruning rates
    parser.add_argument('--prune_rate_init_layer', default=0.8, type=float, help='Unstructured pruning rate for init layer (conv1)')
    parser.add_argument('--prune_rate_layer1', default=0.8, type=float, help='Unstructured pruning rate for layer1 (blocks)')
    parser.add_argument('--prune_rate_layer2', default=0.5, type=float, help='Grouped unstructured pruning rate for layer2 (blocks) with 4-channel grouping')
    parser.add_argument('--prune_rate_layer3', default=0.3, type=float, help='Grouped unstructured pruning rate for layer3 (blocks) with 16-channel grouping')
    
    # NEW: Specific L1-structured (channel-wise) pruning rates for different stages
    # Default values set as per your request: stage1 (16ch) = 4/16=0.25, stage2 (32ch) = 4/32=0.125, stage3 (64ch) = 0
    parser.add_argument('--prune_rate_structured_init_layer', default=0.0, type=float,
                        help='L1-structured (channel-wise) pruning rate for initial conv1 layer. Default 0.0, as covered by stage1.')
    parser.add_argument('--prune_rate_structured_stage1', default=0.25, type=float,
                        help='L1-structured (channel-wise) pruning rate for stage1 (16 channels) layers. (e.g., conv1, layer1.x.convX)')
    parser.add_argument('--prune_rate_structured_stage2', default=0.125, type=float,
                        help='L1-structured (channel-wise) pruning rate for stage2 (32 channels) layers. (e.g., layer2.x.convX)')
    parser.add_argument('--prune_rate_structured_stage3', default=0.0, type=float,
                        help='L1-structured (channel-wise) pruning rate for stage3 (64 channels) layers. (e.g., layer3.x.convX)')
    
    # Updated default for fine-tuning epochs
    parser.add_argument('--finetune_epochs', default=30, type=int, help='Number of epochs for fine-tuning after pruning')
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