import argparse
import time
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# Assuming your resnet20 model definition is in model/resnet.py
from model.resnet import resnet20

# Define AvgMeter utility class
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
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        print("Saving latest checkpoint for resuming...")
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

# ==============================================================================
# Main Function
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pre-training for ResNet20')
    parser.add_argument('--run_name', default='ResNet20_CIFAR10', type=str, help='A name for the training run')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for data loaders')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from the latest checkpoint in a directory')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to the checkpoint directory to resume from.')
    args = parser.parse_args()

    # --- Setup Save Directory ---
    # Create a new directory for a new run
    time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join('checkpoint', f'{time_stamp}_{args.run_name}')
    os.makedirs(save_dir, exist_ok=True)

    if args.resume:
        if not os.path.exists(args.resume_path):
            parser.error(f"Checkpoint file not found in resume directory: {args.resume_path}")
    else:
        args.resume_path = None

    print(f"All outputs will be saved to: {save_dir}")

    # 1. Load Data
    train_loader, test_loader = load_data(args)

    # 2. Instantiate Model
    print("\n--- Initializing ResNet20 model ---")
    net = resnet20()
    net = net.to(device)

    start_epoch = 0
    if args.resume and args.resume_path:
        print(f'==> Resuming from checkpoint: {args.resume_path}..')
        checkpoint = torch.load(args.resume_path, map_location=device)

        state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # 移除 'module.' 前缀
            else:
                new_state_dict[k] = v # 保持原样

        net.load_state_dict(new_state_dict)

        # net.load_state_dict(checkpoint)
        # net.load_state_dict(checkpoint['net'])
        # start_epoch = checkpoint['epoch'] + 1
        # print(f"Resumed model state from epoch {checkpoint['epoch']}. Starting at epoch {start_epoch}.")

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 3. Train the model
    trained_net = train_model(net, train_loader, args, test_loader, epoch_start_val=start_epoch, save_dir=save_dir)

    # 4. Evaluate Final Model Performance
    print("\n--- Evaluating Final Best Model ---")
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        # Load the best performing model for the final evaluation
        final_model = resnet20().to(device)
        final_model.load_state_dict(torch.load(best_model_path, map_location=device))
        if device == 'cuda':
            final_model = torch.nn.DataParallel(final_model)
        final_loss, final_acc = test_model(final_model, test_loader, "Final Best Model Test")
        print(f"\n--- Pre-training Process Complete ---")
        print(f"Final Best Model Accuracy: {final_acc * 100:.2f}%")
    else:
        print("Could not find best_model.pth. The final accuracy is from the last epoch.")