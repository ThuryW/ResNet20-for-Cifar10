import argparse
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# Assuming your resnet20 model definition is in model/resnet.py
from model.resnet import resnet20 

# Define AvgMeter utility class (copied directly for completeness)
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
# Training Function
# ==============================================================================
def train_model(model, train_loader, args, test_loader_for_eval, epoch_start_val=0):
    """
    Trains the model for specified epochs.
    """
    print(f"\n--- Starting training for {args.epochs} epochs ---")
    model.train() # Set to training mode
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0 # Track best accuracy for saving
    
    for epoch in range(epoch_start_val, args.epochs):
        print(f'Train Epoch: {epoch+1}/{args.epochs}')
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

                pbar.set_postfix({'loss': loss_meter.avg, 'acc': acc_meter.avg * 100})
                pbar.update(1)

        scheduler.step()
        print(f"Train Epoch {epoch+1} finished. Loss: {loss_meter.avg:.4f}, Acc: {acc_meter.avg * 100:.2f}%")
        
        # Evaluate after each epoch to find best model
        current_loss, current_acc = test_model(model, test_loader_for_eval, f"Epoch {epoch+1} Test") # Pass test_loader_for_eval here
        
        # Save checkpoint if it's the best model so far
        if current_acc > best_acc:
            print(f"Saving best model with accuracy: {current_acc * 100:.2f}%")
            state = {
                'net': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'acc': current_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.makedirs('checkpoint')
            torch.save(state, f'./checkpoint/{args.epochs}_ckpt_{current_acc*100:.2f}.pth')
            best_acc = current_acc

    print("--- Training complete ---")
    return model

# ==============================================================================
# Test Function
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
# Main Function
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pre-training for ResNet20')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size for data loaders')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epochs', default=200, type=int, help='number of epochs to train')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    # New argument for specifying resume path
    parser.add_argument('--resume_path', type=str, default=None, 
                        help='Path to the .pth checkpoint file to resume from. Required if --resume is used.')
    args = parser.parse_args()

    # Input validation for resume
    if args.resume and args.resume_path is None:
        parser.error("--resume requires --resume_path to be specified.")

    # 1. Load Data
    train_loader, test_loader = load_data(args)

    # 2. Instantiate Model
    print("\n--- Initializing ResNet20 model ---")
    net = resnet20()
    net = net.to(device)

    start_epoch = 0
    if args.resume:
        # Load checkpoint from specified path
        print(f'==> Resuming from checkpoint: {args.resume_path}..')
        if not os.path.exists(args.resume_path):
            print(f'Error: Checkpoint file not found at {args.resume_path}!')
            exit() # Exit if file not found
        
        checkpoint = torch.load(args.resume_path)
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} with accuracy {checkpoint['acc'] * 100:.2f}%")

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 3. Train the model
    # Pass test_loader to train_model for per-epoch evaluation
    trained_net = train_model(net, train_loader, args, test_loader, epoch_start_val=start_epoch)

    # 4. Evaluate Final Trained Model Performance
    print("\n--- Evaluating Final Trained Model ---")
    final_loss, final_acc = test_model(trained_net, test_loader, "Final Trained Model Test")

    print("\n--- Pre-training Process Complete ---")
    print(f"Final Trained Model Accuracy: {final_acc * 100:.2f}%")