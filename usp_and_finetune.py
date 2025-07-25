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
# NEW: Global Pruning Functions
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
            # If you want to prune biases too, uncomment the following:
            # if hasattr(module, 'bias') and module.bias is not None:
            #     parameters_to_prune.append((module, 'bias'))

    # Apply global unstructured pruning based on the specified amount
    # This will determine a global threshold and apply it to all specified parameters
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=global_sparsity_target,
    )

    print("  Global unstructured pruning applied. Performing internal sparsity check...")
    for module, name in parameters_to_prune:
        # Access the pruned weight through the reparametrization
        weight = getattr(module, name) # This will be the masked weight
        num_zeros = torch.sum(weight == 0).item()
        total_elements = weight.numel()
        if total_elements > 0:
            sparsity_percentage = (num_zeros / total_elements) * 100
            print(f"    Layer: {module}, Parameter: {name}, Sparsity: {sparsity_percentage:.2f}% ({num_zeros}/{total_elements})")
    return model

# Note: Structured global pruning is more complex because `prune.ln_structured` operates on a single module at a time
# and there isn't a direct `prune.global_structured` utility in PyTorch for arbitrary layers/dimensions.
# If you need global structured pruning, you would typically:
# 1. Collect the L1 norms (or other importance scores) of all *channels* across all Conv2d layers.
# 2. Flatten these scores into a single list.
# 3. Find the global threshold for channels that achieves the desired sparsity.
# 4. Apply `prune.ln_structured` to each Conv2d layer with a calculated amount based on this global threshold.
# This would require more manual calculation of `amount` per layer based on global statistics.
# For simplicity, if global structured pruning is desired, it often involves custom logic
# to calculate layer-specific `amount`s or using `prune.remove()` and then manually zeroing out channels
# based on a global rank.

# Your original structured pruning functions can still be used if you want to apply
# a fixed percentage to *each* layer's channels, but that's not "global target sparsity".
# For the purpose of achieving an overall sparsity with layer-wise varying rates,
# global unstructured pruning is the most straightforward approach with `torch.nn.utils.prune`.


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning and Fine-tuning')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs to fine-tune')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', default=None, type=str, help='path to latest checkpoint (if resume is true)')
    parser.add_argument('--pretrained_model_path', default='/home/wangtianyu/my_resnet20/base_models/20_ckpt_92.23.pth', type=str, help='path to the pretrained model checkpoint')
    
    # NEW ARGUMENT for GLOBAL sparsity
    parser.add_argument('--global_unstructured_sparsity', default=0.8, type=float, 
                        help='Target global unstructured sparsity for the model (e.g., 0.8 for 80% total zeros)')
    # You can keep this if you want to combine it with global unstructured,
    # but it will apply a fixed percentage per layer, not contribute to global structured sparsity.
    parser.add_argument('--structured_pruning_amount_per_layer', default=0.0, type=float, 
                        help='Amount for structured pruning applied individually to each conv layer (0.0 means no structured pruning)')
    
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

    # Apply individual layer structured pruning if specified (this is not globally controlled)
    # model = structured_prune_conv_layers(model, amount=args.structured_pruning_amount_per_layer)
    
    # NEW: Apply Global Unstructured Pruning
    model = global_unstructured_prune(model, global_sparsity_target=args.global_unstructured_sparsity)

    print("\nVerifying pruning masks after application (before fine-tuning):")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                print(f"  Module {name} is pruned. Active weight sparsity: {(torch.sum(module.weight == 0).item() / module.weight.numel()) * 100:.2f}%")
    
    if device == 'cuda':
        print("\nWrapping model in DataParallel for fine-tuning...")
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    finetuned_model = train_model(model, train_loader, args, test_loader, save_dir=finetuned_save_dir)

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
    
    # Apply dummy pruning to final_model before loading state_dict to handle _orig/_mask
    print("Applying dummy pruning to final_model to prepare for loading state_dict...")
    for name, module in final_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Use a tiny amount for dummy pruning to ensure the _orig and _mask parameters are created.
            prune.l1_unstructured(module, name='weight', amount=0.0001) 
    print("Dummy pruning applied to final_model.")

    final_model.load_state_dict(new_state_dict_finetuned)
    print("Fine-tuned model state_dict loaded successfully into final_model.")

    print("\n--- Permanently applying pruning to final_model (prune.remove()) ---")
    for name, module in final_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
    print("Pruning permanently applied to final_model.")

    if args.save_final_pruned_model_path:
        os.makedirs(os.path.dirname(args.save_final_pruned_model_path), exist_ok=True)
        torch.save(final_model.state_dict(), args.save_final_pruned_model_path)
        print(f"\nFinal pruned model (weights solidified) saved to: {args.save_final_pruned_model_path}")

    print("\nVerifying sparsity of the loaded and solidified final model:")
    compute_total_sparsity(final_model)
    
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