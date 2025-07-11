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

from model.resnet import resnet20 # 确保你的resnet20模型在这里定义
from utils import AvgMeter # 确保你的AvgMeter工具函数在这里定义

# 配置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==============================================================================
# 数据加载函数
# ==============================================================================
def load_data(args):
    """
    加载 CIFAR-10 训练集和测试集。
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
# 剪枝函数
# ==============================================================================
def prune_model(model, pruning_amount):
    """
    对模型进行非结构化剪枝。
    使用 L1 非结构化剪枝方法，即根据权重绝对值大小进行剪枝。
    """
    print(f"\n--- Applying unstructured pruning with amount: {pruning_amount * 100:.2f}% ---")

    # 遍历模型的所有模块（层）
    for name, module in model.named_modules():
        # 我们只对 Conv2d 和 Linear 层的权重进行剪枝
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)

    # 打印剪枝后的稀疏度
    total_elements = 0
    zero_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight_mask'): 
                total_elements += module.weight_mask.nelement()
                zero_elements += torch.sum(module.weight_mask == 0)
    actual_sparsity = (zero_elements / total_elements) * 100 if total_elements > 0 else 0
    print(f"Model sparsity after pruning: {actual_sparsity:.2f}%")

    return model

# ==============================================================================
# 微调函数
# ==============================================================================
def finetune_model(model, train_loader, args):
    """
    对剪枝后的模型进行微调。
    """
    print(f"\n--- Starting fine-tuning for {args.finetune_epochs} epochs ---")
    model.train() # 设置为训练模式
    
    # 定义优化器和损失函数
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

                pbar.set_postfix({'loss': loss_meter.avg(), 'acc': acc_meter.avg() * 100})
                pbar.update(1)

        scheduler.step()
        print(f"Finetune Epoch {epoch+1} finished. Loss: {loss_meter.avg():.4f}, Acc: {acc_meter.avg() * 100:.2f}%")
    
    print("--- Fine-tuning complete ---")
    return model

# ==============================================================================
# 测试函数 (与你之前的基本相同，但统一了模型加载逻辑)
# ==============================================================================
def test_model(model, test_loader, description="Test"):
    """
    评估模型的性能。
    """
    model.eval() # 设置为评估模式
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
    test_loss = loss_meter.avg()
    test_acc = acc_meter.avg()

    print(f"--- {description} Result --- Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

# ==============================================================================
# 保存模型函数
# ==============================================================================
def save_model(model, args, suffix="pruned_finetuned"):
    """
    保存剪枝微调后的模型。
    """
    save_dir = 'pruned_checkpoints'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # 移除剪枝相关的钩子和参数，将稀疏权重固化为实际权重
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)) and prune.is_pruned(module):
            # 将 prune.remove_pruning 改为 prune.remove
            prune.remove(module, 'weight')
    
    # 如果模型是 DataParallel 包装的，保存其内部的 .module
    if isinstance(model, nn.DataParallel):
        state = {
            'net': model.module.state_dict(),
            'pruning_amount': args.prune_amount,
            'finetune_epochs': args.finetune_epochs,
            'test_acc': args.final_test_acc # 使用args.final_test_acc，因为test_acc不是全局变量
        }
    else:
        state = {
            'net': model.state_dict(),
            'pruning_amount': args.prune_amount,
            'finetune_epochs': args.finetune_epochs,
            'test_acc': args.final_test_acc # 使用args.final_test_acc
        }
    
    # 构造保存文件名
    filename = f"resnet20_{suffix}_prune{int(args.prune_amount*100)}_ft{args.finetune_epochs:.0f}.pth" # 确保finetune_epochs格式化为整数
    filepath = os.path.join(save_dir, filename)
    
    torch.save(state, filepath)
    print(f"\nModel saved to: {filepath}")
    
# ==============================================================================
# 主函数
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning and Fine-tuning')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size for data loaders')
    parser.add_argument('--model_path', default='./checkpoint/20_ckpt_92.23.pth', type=str, help='path to original trained model checkpoint')
    parser.add_argument('--prune_amount', default=0.5, type=float, help='Amount of unstructured pruning (0.0 to 1.0)')
    parser.add_argument('--finetune_epochs', default=10, type=int, help='Number of epochs for fine-tuning after pruning')
    parser.add_argument('--finetune_lr', default=0.01, type=float, help='Learning rate for fine-tuning')
    args = parser.parse_args()

    # 1. 加载数据
    train_loader, test_loader = load_data(args)

    # 2. 加载预训练模型
    print(f"\n--- Loading pre-trained model from {args.model_path} ---")
    net = resnet20()
    net = net.to(device)

    # 加载 checkpoint
    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['net']
    
    # 处理 state_dict 的键，移除 'module.' 前缀（如果存在）
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)

    # 如果要使用多GPU，在加载权重后包装
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # 3. 评估原始模型性能
    print("\n--- Evaluating Original Model ---")
    original_loss, original_acc = test_model(net, test_loader, "Original Model Test")

    # 4. 应用剪枝
    pruned_net = prune_model(net, args.prune_amount) # 注意：这里prune_model会修改传入的net对象

    # 5. 评估剪枝后（未微调）模型性能
    print("\n--- Evaluating Pruned (before finetune) Model ---")
    pruned_loss, pruned_acc = test_model(pruned_net, test_loader, "Pruned Model Test")

    # 6. 微调剪枝后的模型
    finetuned_net = finetune_model(pruned_net, train_loader, args)

    # 7. 评估微调后模型性能
    print("\n--- Evaluating Fine-tuned Model ---")
    final_loss, final_acc = test_model(finetuned_net, test_loader, "Fine-tuned Model Test")
    test_acc = final_acc # 将最终精度赋值给test_acc，用于保存
    # 将最终精度存储到args中，以便save_model函数访问
    args.final_test_acc = final_acc 

    # 8. 保存微调后的模型
    save_model(finetuned_net, args, suffix="pruned_finetuned")

    print("\n--- Pruning and Fine-tuning Process Complete ---")
    print(f"Original Accuracy: {original_acc * 100:.2f}%")
    print(f"Pruned (no finetune) Accuracy: {pruned_acc * 100:.2f}%")
    print(f"Fine-tuned Accuracy: {final_acc * 100:.2f}%")