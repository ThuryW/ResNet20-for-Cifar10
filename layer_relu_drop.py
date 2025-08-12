import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
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
alpha = 0.5  # CE loss 权重
temperature = 4.0
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

# ==== 知识蒸馏 loss ====
def kd_loss_fn(outputs, teacher_outputs, labels, alpha, T):
    ce_loss = F.cross_entropy(outputs, labels)
    kd_loss = F.kl_div(F.log_softmax(outputs / T, dim=1),
                       F.softmax(teacher_outputs / T, dim=1),
                       reduction='batchmean') * (T * T)
    return alpha * ce_loss + (1 - alpha) * kd_loss

# ==== 修改模型 ====
def modify_model(case):
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

    # Stage1: 去掉所有 ReLU
    for block in model.layer1:
        block.relu1 = nn.Identity()
        block.relu2 = nn.Identity()

    # Stage2 & Stage3: 每个 block 去掉一个 ReLU
    for stage in [model.layer2, model.layer3]:
        for block in stage:
            if case == "caseA":  # 去掉 relu1
                block.relu1 = nn.Identity()
            elif case == "caseB":  # 去掉 relu2
                block.relu2 = nn.Identity()

    return model

# ==== 蒸馏训练 ====
def train_kd(case):
    # Teacher
    teacher = resnet20().to(device)

    if device == 'cuda':
        teacher = torch.nn.DataParallel(teacher)
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
    teacher.load_state_dict(new_state_dict)

    # teacher_ckpt = torch.load(base_model_path, map_location=device)
    # teacher.load_state_dict(teacher_ckpt)
    teacher.eval()

    # Student
    student = modify_model(case)

    # 保存路径
    start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    save_dir = os.path.join(save_root, start_time_str, case)
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pth")
    latest_path = os.path.join(save_dir, "latest.pth")
    log_path = os.path.join(save_dir, "train_log.txt")

    # 初始精度
    init_acc = test(student)

    optimizer = torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    log_lines = [f"Case={case}, Init Acc={init_acc:.2f}%"]
    best_acc = 0.0

    for epoch in range(finetune_epochs):
        student.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
            student_outputs = student(inputs)
            loss = kd_loss_fn(student_outputs, teacher_outputs, targets, alpha, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc = test(student)
        log_lines.append(f"Epoch [{epoch+1}/{finetune_epochs}] Acc={acc:.2f}%")
        print(log_lines[-1])
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), best_path)

    torch.save(student.state_dict(), latest_path)
    log_lines.append(f"Best Acc={best_acc:.2f}%")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"{case} 训练完成，best/latest/log 保存到 {save_dir}")

# ==== 主流程 ====
for case in ["caseA", "caseB"]:
    train_kd(case)
