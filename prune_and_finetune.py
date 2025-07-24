import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from model.resnet import resnet20
from train import train_model, test_model, load_data # Assuming 'train' module exists and contains these

# Import datetime for timestamped directories
from datetime import datetime
import argparse # Import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Loads CIFAR-10 training and test datasets."""
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader

# ==============================================================================
# Training Function
# ==============================================================================
def train_model(model, train_loader, args, test_loader_for_eval, epoch_start_val=0, save_dir=None):
    """Trains the model for specified epochs."""
    print(f"\n--- Starting training for {args.epochs} total epochs ---")
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    
    # If resuming, load optimizer and scheduler states
    if args.resume and args.resume_path:
        print(f"Loading optimizer and scheduler states from {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint.get('acc', 0.0)
            print(f"Resumed optimizer, scheduler, and best_acc: {best_acc * 100:.2f}%")
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
        
        # Evaluate after each epoch
        current_loss, current_acc = test_model(model, test_loader_for_eval, f"Epoch {epoch+1} Test")
        
        # Save best model and latest checkpoint
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
            print(f"New best accuracy: {best_acc * 100:.2f}%. Saving best model...")
            # Ensure the directory exists before saving
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print("Saving latest checkpoint for resuming...")
        # Ensure the directory exists before saving
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

# =====================
# 通道剪枝（结构化）
# =====================
def structured_prune_conv_layers(model, amount=0.5):
    print(f"\n==> Applying structured channel pruning (L1 norm) with amount={amount}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            prune.remove(module, 'weight')
    return model


# =====================
# 非结构化剪枝
# =====================
def unstructured_prune_weights(model, amount=0.5):
    print(f"\n==> Applying unstructured weight pruning (L1 norm) with amount={amount}")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model


# =====================
# 剪枝率计算
# =====================
def compute_total_sparsity(model):
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    sparsity = zero_params / total_params
    print(f"==> Total sparsity: {sparsity * 100:.2f}%")
    return sparsity


# =====================
# 主流程
# =====================
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning and Fine-tuning')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=60, type=int, help='number of epochs to fine-tune')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', default=None, type=str, help='path to latest checkpoint (if resume is true)')
    parser.add_argument('--pretrained_model_path', default='/home/wangtianyu/my_resnet20/checkpoint/20_ckpt_92.23.pth', type=str, help='path to the pretrained model checkpoint')
    parser.add_argument('--structured_pruning_amount', default=0.5, type=float, help='amount for structured pruning')
    parser.add_argument('--unstructured_pruning_amount', default=0.8, type=float, help='amount for unstructured pruning')
    parser.add_argument('--min_acc_drop', default=0.01, type=float, help='maximum allowed accuracy drop after pruning and finetuning')
    
    args = parser.parse_args()

    # 1. Load Data
    train_loader, test_loader = load_data(args)

    # 2. Load Pretrained ResNet20
    model = resnet20().to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    print(f"Loading pretrained model from: {args.pretrained_model_path}")
    checkpoint = torch.load(args.pretrained_model_path, map_location=device)
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v # 保持原样
    # If the model itself is DataParallel, load state dict into its 'module' attribute
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    model.eval()

    # 3. Evaluate original accuracy
    _, base_acc = test_model(model, test_loader, description="Original Model")

    # 4. Apply Structured Pruning
    # Ensure the model is unwrapped from DataParallel before pruning
    if isinstance(model, nn.DataParallel):
        model = model.module
    model = structured_prune_conv_layers(model, amount=args.structured_pruning_amount)

    # 5. Apply Unstructured Pruning
    model = unstructured_prune_weights(model, amount=args.unstructured_pruning_amount)

    # 6. Calculate sparsity
    sparsity = compute_total_sparsity(model)
    # assert sparsity >= 0.90, "Sparsity not high enough!"

    # 7. Fine-tune the pruned model
    # Wrap the model in DataParallel again for training if using CUDA
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    # Create a timestamped directory for saving finetuned models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_save_dir = os.path.join('./autorun', timestamp)
    os.makedirs(finetuned_save_dir, exist_ok=True)
    print(f"Fine-tuned model checkpoints will be saved in: {finetuned_save_dir}")

    finetuned_model = train_model(model, train_loader, args, test_loader, save_dir=finetuned_save_dir)

    # 8. Evaluate finetuned model
    final_model = resnet20().to(device)
    
    # Load the state dict from the best_model.pth in the timestamped directory
    finetuned_checkpoint_path = os.path.join(finetuned_save_dir, 'best_model.pth')
    if not os.path.exists(finetuned_checkpoint_path):
        print(f"Warning: best_model.pth not found in {finetuned_save_dir}. Attempting to load latest_checkpoint.pth instead.")
        finetuned_checkpoint_path = os.path.join(finetuned_save_dir, 'latest_checkpoint.pth')
        if not os.path.exists(finetuned_checkpoint_path):
            raise FileNotFoundError(f"Neither best_model.pth nor latest_checkpoint.pth found in {finetuned_save_dir}")
        else:
            print("Loading latest_checkpoint.pth for final evaluation.")
            checkpoint_finetuned = torch.load(finetuned_checkpoint_path, map_location=device)
            state_dict_finetuned = checkpoint_finetuned['net']
    else:
        print(f"Loading best_model.pth from {finetuned_save_dir} for final evaluation.")
        state_dict_finetuned = torch.load(finetuned_checkpoint_path, map_location=device)

    # Apply the same state_dict processing as the initial model loading
    new_state_dict_finetuned = {}
    for k, v in state_dict_finetuned.items():
        if k.startswith('module.'):
            new_state_dict_finetuned[k[7:]] = v # Remove 'module.' prefix
        else:
            new_state_dict_finetuned[k] = v # Keep as is
            
    # Load state dict into the final_model (which is not yet DataParallel)
    final_model.load_state_dict(new_state_dict_finetuned)
    
    # Wrap the final_model in DataParallel for evaluation if using CUDA
    if device == 'cuda':
        final_model = torch.nn.DataParallel(final_model)
        cudnn.benchmark = True

    _, final_acc = test_model(final_model, test_loader, description="Pruned & Finetuned Model")

    # 9. Accuracy Check
    acc_drop = base_acc - final_acc
    print(f"\n==> Accuracy drop: {acc_drop * 100:.2f}%")
    # assert acc_drop < args.min_acc_drop, f"Accuracy drop ({acc_drop * 100:.2f}%) is too high! (Allowed: <{args.min_acc_drop * 100:.2f}%)"

    print("\nPruning + Fine-tuning Complete!")


if __name__ == '__main__':
    main()