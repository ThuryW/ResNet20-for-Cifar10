import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms

from datetime import datetime
import argparse
import numpy as np # Import numpy for percentile calculation

try:
    from model.resnet import resnet20
except ImportError:
    print("Error: Could not import resnet20 from model/resnet.py.")
    print("Please ensure you have a 'model' directory in the same location as this script,")
    print("and that 'resnet.py' exists within it and defines a 'resnet20' function.")
    exit(1)

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

def load_data(args):
    """Loads CIFAR-10 training and test datasets."""
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader

def train_model(model, train_loader, args, test_loader_for_eval, epoch_start_val=0, save_dir=None):
    """Trains the model for specified epochs."""
    print(f"\n--- Starting training for {args.epochs} total epochs ---")
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=(args.epochs//3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    
    if args.resume and args.resume_path:
        print(f"Loading optimizer and scheduler states from {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint.get('acc', 0.0)
            epoch_start_val = checkpoint.get('epoch', 0) + 1
            print(f"Resumed optimizer, scheduler, and best_acc: {best_acc * 100:.2f}% from epoch {epoch_start_val-1}")
        else:
            print("Optimizer/Scheduler state not found in checkpoint. Starting fresh.")


    for epoch in range(epoch_start_val, args.epochs):
        print(f'\nTrain Epoch: {epoch+1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')
        loss_meter = AvgMeter()
        acc_meter = AvgMeter()

        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}") as pbar:
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

                pbar.set_postfix({'loss': loss_meter.avg, 'acc': f'{acc_meter.avg * 100:.2f}%'})
                pbar.update(1)

        scheduler.step()
        
        current_loss, current_acc = test_model(model, test_loader_for_eval, f"Epoch {epoch+1} Test")
        
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
            print(f"New best accuracy: {best_acc * 100:.2f}%. Saving best model...")
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                       os.path.join(save_dir, 'best_model.pth'))

        print("Saving latest checkpoint for resuming...")
        os.makedirs(save_dir, exist_ok=True)
        state = {
            'net': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'acc': current_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, os.path.join(save_dir, 'latest_checkpoint.pth'))

    print("--- Training complete ---")
    return model

def test_model(model, test_loader, description="Test"):
    """Evaluates model performance."""
    model.eval()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Starting {description} ---")
    with torch.no_grad():
        for image_batch, gt_batch in tqdm(test_loader, desc=description):
            image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
            pred_batch = model(image_batch)
            loss = criterion(pred_batch, gt_batch.long())
            loss_meter.add(loss.item(), image_batch.size(0))
            acc = (pred_batch.argmax(dim=-1).long() == gt_batch).float().mean()
            acc_meter.add(acc.item(), image_batch.size(0))

    test_loss = loss_meter.avg
    test_acc = acc_meter.avg

    print(f"--- {description} Result --- Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

def compute_total_sparsity(model):
    total_prunable_elements = 0
    zero_prunable_elements = 0
    
    print("\n==> Layer-wise sparsity:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                current_weight = module.weight
                
                current_total_elements = current_weight.numel()
                current_zero_elements = torch.sum(current_weight == 0).item()

                total_prunable_elements += current_total_elements
                zero_prunable_elements += current_zero_elements

                layer_sparsity = (current_zero_elements / current_total_elements) * 100 if current_total_elements > 0 else 0
                print(f"  {name}.weight: {layer_sparsity:.2f}% zeroed")
        
    if total_prunable_elements > 0:
        overall_prunable_sparsity = zero_prunable_elements / total_prunable_elements
        print(f"\n==> Overall sparsity for PRUNABLE weights (Conv2d & Linear): {overall_prunable_sparsity * 100:.2f}%")
    else:
        print("\nNo prunable weights (Conv2d & Linear) found to calculate specific sparsity, or they have 0 elements.")

    total_all_params = 0
    zero_all_params = 0
    for param in model.parameters():
        total_all_params += param.numel()
        zero_all_params += torch.sum(param == 0).item()
        
    overall_model_sparsity = zero_all_params / total_all_params if total_all_params > 0 else 0
    print(f"==> Total model sparsity (including ALL parameters): {overall_model_sparsity * 100:.2f}%")
    return overall_model_sparsity

# ==============================================================================
# Global Pruning Functions (Unstructured and NEW Structured)
# ==============================================================================

def global_unstructured_prune(model, global_sparsity_target=0.8):
    """
    Applies unstructured pruning globally across all prunable layers (Conv2d, Linear)
    to achieve a target sparsity for the entire model's prunable weights.
    """
    print(f"\n==> Applying global unstructured pruning with target sparsity={global_sparsity_target}")
    if global_sparsity_target == 0:
        print("  Skipping global unstructured pruning as target sparsity is 0.")
        return model

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=global_sparsity_target,
    )

    print("  Global unstructured pruning applied. Performing internal sparsity check...")
    for module, name in parameters_to_prune:
        weight = getattr(module, name)
        num_zeros = torch.sum(weight == 0).item()
        total_elements = weight.numel()
        if total_elements > 0:
            sparsity_percentage = (num_zeros / total_elements) * 100
            print(f"    Layer: {module}, Parameter: {name}, Sparsity: {sparsity_percentage:.2f}% ({num_zeros}/{total_elements})")
    return model

def global_structured_prune_channels(model, global_sparsity_target=0.5):
    """
    Applies structured (channel) pruning globally across all Conv2d layers
    to achieve a target sparsity for the entire model's convolutional output channels.
    This implementation involves directly zeroing out channels after determining a global threshold.
    """
    print(f"\n==> Applying global structured channel pruning with target sparsity={global_sparsity_target}")
    if global_sparsity_target == 0:
        print("  Skipping global structured pruning as target sparsity is 0.")
        return model

    all_channel_l1_norms = []
    prunable_conv_layers = []

    # Collect L1 norms of output channels for all Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate L1 norm for each output channel (dim=0)
            # This sums the absolute values across input channels, kernel height, and kernel width
            l1_norms = torch.norm(module.weight.data, p=1, dim=(1, 2, 3)) # (out_channels, in_channels, kH, kW)
            all_channel_l1_norms.extend(l1_norms.detach().cpu().numpy())
            prunable_conv_layers.append((name, module))

    if not all_channel_l1_norms:
        print("  No Conv2d layers found for structured pruning or no channels to prune.")
        return model

    # Calculate the global threshold
    # Sort the L1 norms and find the value at the (global_sparsity_target) percentile
    threshold_idx = int(len(all_channel_l1_norms) * global_sparsity_target)
    # Ensure threshold_idx is within bounds
    threshold_idx = min(threshold_idx, len(all_channel_l1_norms) - 1)
    
    # Use np.partition for efficiency if the list is large, otherwise sort is fine
    # threshold = np.sort(all_channel_l1_norms)[threshold_idx]
    threshold = np.partition(all_channel_l1_norms, threshold_idx)[threshold_idx]

    print(f"  Calculated global L1 norm threshold for channels: {threshold:.6f}")

    # Apply pruning based on the global threshold
    total_channels_pruned = 0
    total_channels_in_model = 0

    for name, module in prunable_conv_layers:
        l1_norms = torch.norm(module.weight.data, p=1, dim=(1, 2, 3))
        
        # Identify channels to prune (those whose L1 norm is below the global threshold)
        channels_to_prune_mask = l1_norms <= threshold
        
        # Get the indices of channels to prune
        channels_to_prune_indices = torch.nonzero(channels_to_prune_mask, as_tuple=True)[0]

        num_channels_pruned_in_layer = len(channels_to_prune_indices)
        total_channels_in_layer = module.weight.shape[0] # output channels
        
        # If no channels to prune in this layer, continue
        if num_channels_pruned_in_layer == 0:
            print(f"    Layer: {name} - No channels pruned (0/{total_channels_in_layer})")
            continue

        # Manually set the weights of these channels to zero
        # This effectively prunes the output channels
        with torch.no_grad():
            module.weight[channels_to_prune_indices] = 0.0

        total_channels_pruned += num_channels_pruned_in_layer
        total_channels_in_model += total_channels_in_layer

        layer_pruning_percentage = (num_channels_pruned_in_layer / total_channels_in_layer) * 100
        print(f"    Layer: {name}, Channels Pruned: {num_channels_pruned_in_layer}/{total_channels_in_layer} ({layer_pruning_percentage:.2f}%)")
    
    overall_structured_sparsity = (total_channels_pruned / total_channels_in_model) * 100 if total_channels_in_model > 0 else 0
    print(f"\n  Global structured pruning applied. Overall channel sparsity: {overall_structured_sparsity:.2f}% ({total_channels_pruned}/{total_channels_in_model} channels).")

    # Important: Since we manually zeroed out, there are no _orig and _mask.
    # So, `prune.is_pruned()` will return False, and `prune.remove()` is not needed for these.
    # The `compute_total_sparsity` function will correctly count these zeros.

    return model

# Override the structured_prune_conv_layers with a placeholder or remove it,
# as the new global structured pruning function will handle it.
# def structured_prune_conv_layers(model, amount=0.5):
#     print(f"\n==> Applying structured channel pruning (L1 norm) with amount={amount}")
#     if amount == 0:
#         print("  Skipping structured pruning as amount is 0.")
#         return model
#
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
#
#     print("  Structured pruning applied. Performing internal sparsity check...")
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Conv2d):
#             weight = getattr(module, 'weight')
#             num_zeros = torch.sum(weight == 0).item()
#             total_elements = weight.numel()
#             if total_elements > 0:
#                 sparsity_percentage = (num_zeros / total_elements) * 100
#                 print(f"    Layer: {name}, Structured Weight Sparsity: {sparsity_percentage:.2f}% ({num_zeros}/{total_elements})")
#     return model


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning and Fine-tuning')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs to fine-tune')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', default=None, type=str, help='path to latest checkpoint (if resume is true)')
    parser.add_argument('--pretrained_model_path', default='/home/wangtianyu/my_resnet20/base_models/20_ckpt_92.23.pth', type=str, help='path to the pretrained model checkpoint')
    
    # Global pruning arguments
    parser.add_argument('--global_unstructured_sparsity', default=0.7, type=float, # Changed default to 0.0
                        help='Target global unstructured sparsity for the model (e.g., 0.8 for 80% total zeros)')
    parser.add_argument('--global_structured_sparsity', default=0.05, type=float, # NEW ARGUMENT
                        help='Target global structured (channel) sparsity for Conv2d layers (e.g., 0.5 for 50% channels zeroed)')
    
    parser.add_argument('--min_acc_drop', default=0.01, type=float, help='maximum allowed accuracy drop after pruning and finetuning')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fine-tuned model checkpoints and logs')
    parser.add_argument('--save_final_pruned_model_path', type=str, default=None, 
                        help='Optional: Path to save the final fine-tuned model after permanently applying pruning (removing _orig/_mask)')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(finetuned_save_dir, exist_ok=True)
    print(f"All outputs (checkpoints, logs) will be saved in: {finetuned_save_dir}")

    train_loader, test_loader = load_data(args)

    model = resnet20().to(device)
    
    print(f"\nLoading pretrained model from: {args.pretrained_model_path}")
    try:
        checkpoint = torch.load(args.pretrained_model_path, map_location=device)
        state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        print("Pretrained model loaded successfully.")
    except Exception as e:
        print(f"Error loading pretrained model from {args.pretrained_model_path}: {e}")
        print("Please ensure the path is correct and the file is a valid PyTorch checkpoint.")
        exit()

    model.eval()

    _, base_acc = test_model(model, test_loader, description="Original Model")

    if isinstance(model, nn.DataParallel):
        print("Unwrapping model from DataParallel for pruning...")
        model = model.module

    # Apply Global Structured Pruning first (as it directly modifies weights)
    # This comes before unstructured pruning because structured pruning affects the 'shape' of weights
    # (by making channels zero, which then influences unstructured pruning on remaining non-zero weights).
    model = global_structured_prune_channels(model, global_sparsity_target=args.global_structured_sparsity)
    
    # Apply Global Unstructured Pruning
    model = global_unstructured_prune(model, global_sparsity_target=args.global_unstructured_sparsity)

    print("\nVerifying pruning masks and zeros after all pruning applications (before fine-tuning):")
    # For unstructured pruning, prune.is_pruned() will be True.
    # For structured pruning (manual zeroing), prune.is_pruned() will be False, but compute_total_sparsity will count zeros.
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                # This check is for `torch.nn.utils.prune` applied masks
                print(f"  Module {name} is pruned (via mask). Active weight sparsity: {(torch.sum(module.weight == 0).item() / module.weight.numel()) * 100:.2f}%")
            else:
                # This covers modules where channels were manually zeroed (structured pruning) or not pruned by `torch.nn.utils.prune`
                current_weight = module.weight
                num_zeros = torch.sum(current_weight == 0).item()
                total_elements = current_weight.numel()
                if total_elements > 0:
                    sparsity_percentage = (num_zeros / total_elements) * 100
                    if sparsity_percentage > 0: # Only print if there's actual sparsity
                        print(f"  Module {name} has direct zeros (possibly structured). Active weight sparsity: {sparsity_percentage:.2f}%")

    # Re-wrap the model in DataParallel for training if using CUDA
    if device == 'cuda':
        print("\nWrapping model in DataParallel for fine-tuning...")
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    finetuned_model = train_model(model, train_loader, args, test_loader, save_dir=finetuned_save_dir)

    # --- Final Model Loading and Solidification ---
    final_model = resnet20().to(device)
    
    finetuned_checkpoint_path = os.path.join(finetuned_save_dir, 'best_model.pth')
    state_dict_finetuned = None

    if os.path.exists(finetuned_checkpoint_path):
        print(f"\nLoading best_model.pth from {finetuned_checkpoint_path} for final evaluation.")
        state_dict_finetuned = torch.load(finetuned_checkpoint_path, map_location=device)
    else:
        print(f"\nWarning: best_model.pth not found in {finetuned_save_dir}. Attempting to load latest_checkpoint.pth instead.")
        finetuned_checkpoint_path = os.path.join(finetuned_save_dir, 'latest_checkpoint.pth')
        if os.path.exists(finetuned_checkpoint_path):
            print("Loading latest_checkpoint.pth for final evaluation.")
            checkpoint_data = torch.load(finetuned_checkpoint_path, map_location=device)
            state_dict_finetuned = checkpoint_data['net']
        else:
            raise FileNotFoundError(f"Neither best_model.pth nor latest_checkpoint.pth found in {finetuned_save_dir}")

    new_state_dict_finetuned = {}
    for k, v in state_dict_finetuned.items():
        if k.startswith('module.'):
            new_state_dict_finetuned[k[7:]] = v
        else:
            new_state_dict_finetuned[k] = v
    
    # IMPORTANT: Apply a small dummy unstructured pruning to `final_model` before loading state_dict
    # This ensures that `_orig` and `_mask` parameters are created if the loaded state_dict
    # contains them (which it will if `global_unstructured_prune` was used).
    # Structured pruning (manual zeroing) does not create these, so it's fine.
    print("Applying dummy pruning to final_model to prepare for loading state_dict for unstructured pruning layers...")
    for name, module in final_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Only apply dummy pruning if it was likely unstructured pruned in training
            # A small amount (e.g., 0.0001) is sufficient to create the reparameterization
            prune.l1_unstructured(module, name='weight', amount=0.0001) 
    print("Dummy pruning applied to final_model.")

    final_model.load_state_dict(new_state_dict_finetuned)
    print("Fine-tuned model state_dict loaded successfully into final_model.")

    print("\n--- Permanently applying pruning to final_model (prune.remove()) ---")
    for name, module in final_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                # This removes _orig and _mask for unstructured pruning, making weights permanent.
                prune.remove(module, 'weight')
    print("Pruning permanently applied to final_model.")

    if args.save_final_pruned_model_path:
        os.makedirs(os.path.dirname(args.save_final_pruned_model_path), exist_ok=True)
        # Save the state_dict of the model AFTER prune.remove()
        torch.save(final_model.state_dict(), args.save_final_pruned_model_path)
        print(f"\nFinal pruned model (weights solidified) saved to: {args.save_final_pruned_model_path}")

    print("\nVerifying sparsity of the loaded and solidified final model:")
    compute_total_sparsity(final_model) # This compute_total_sparsity function is compatible with solidified weights
    
    if device == 'cuda':
        final_model = torch.nn.DataParallel(final_model)
        cudnn.benchmark = True

    _, final_acc = test_model(final_model, test_loader, description="Pruned & Finetuned Model (Solidified)")

    acc_drop = base_acc - final_acc
    print(f"\n==> Original Accuracy: {base_acc * 100:.2f}%")
    print(f"==> Pruned & Finetuned (Solidified) Accuracy: {final_acc * 100:.2f}%")
    print(f"==> Absolute Accuracy Drop: {acc_drop * 100:.2f}%")
    
    print("\nPruning + Fine-tuning Complete!")


if __name__ == '__main__':
    main()