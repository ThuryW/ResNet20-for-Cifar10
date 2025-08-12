import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
from model.resnet import resnet20

# ==== 配置 ====
base_model_path = "./base_models/20_ckpt_92.23.pth"
save_root = "./checkpoints/reludrop_ckpt"
finetune_epochs = 20
batch_size = 128
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== CIFAR-10 数据 ====
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

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

# ==== 测试函数 ====
def test(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

# ==== finetune 函数 ====
def finetune_and_save(stage_name):
    # 加载模型
    model = resnet20().to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    checkpoint = torch.load(base_model_path, map_location=device)
    state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint # Handle different checkpoint formats

    # Adjust state_dict keys for potential DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # Remove 'module.' prefix
        else:
            new_state_dict[k] = v # Keep as is
    model.load_state_dict(new_state_dict)

    # 剪掉指定 stage 的所有 ReLU
    stage = getattr(model, stage_name)
    for block in stage:
        for name, module in block.named_children():
            if isinstance(module, nn.ReLU):
                setattr(block, name, nn.Identity())

    # ==== 测试初始精度 ====
    init_acc = test(model)
    print(f"Stage={stage_name} 初始精度（移除 ReLU，未 finetune）: {init_acc:.2f}%")

    # ==== 创建保存目录 ====
    start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_dir = os.path.join(save_root, start_time_str, stage_name)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "finetune_log.txt")
    best_model_path = os.path.join(save_dir, "best.pth")
    latest_model_path = os.path.join(save_dir, "latest.pth")

    # ==== 优化器等 ====
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ==== finetune ====
    best_acc = 0.0
    log_lines = [
        f"Stage={stage_name}, ReLU removed",
        f"Initial Acc (no finetune): {init_acc:.2f}%"
    ]

    for epoch in range(finetune_epochs):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = test(model)
        log_lines.append(f"Epoch [{epoch+1}/{finetune_epochs}] Test Acc: {acc:.2f}%")
        print(log_lines[-1])

        # 保存 best
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_model_path)

    # 保存 latest
    torch.save(model.state_dict(), latest_model_path)
    log_lines.append(f"Best Acc: {best_acc:.2f}%")

    # 保存日志
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    print(f"Stage={stage_name} 完成，best/latest 模型及日志已保存到 {save_dir}")

# ==== 主流程 ====
for stage_name in ["layer1", "layer2", "layer3"]:
    finetune_and_save(stage_name)
