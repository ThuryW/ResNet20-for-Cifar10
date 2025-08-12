import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.resnet import resnet20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 构造修改过的模型（ReLU 移除规则） =====
def build_student_caseB_remove_relus():
    model = resnet20()
    # 移除 conv1 初始 ReLU
    if hasattr(model, "relu_initial"):
        model.relu_initial = nn.Identity()
    # Stage1: 移除所有 ReLU
    for block in model.layer1:
        if hasattr(block, "relu1"):
            block.relu1 = nn.Identity()
        if hasattr(block, "relu2"):
            block.relu2 = nn.Identity()
    # Stage2 & Stage3: 每个 block 移除 relu2 (CaseB)
    for stage in [model.layer2, model.layer3]:
        for block in stage:
            if hasattr(block, "relu2"):
                block.relu2 = nn.Identity()
    return model

# ===== CIFAR-10 数据 =====
def get_testloader(batch_size=100, num_workers=4):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# ===== 测试函数 =====
def evaluate(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.0 * correct / total
    return acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test modified ResNet20 with removed ReLUs")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    # 构建修改过的模型
    model = build_student_caseB_remove_relus()
    # 加载权重
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt)
    model = model.to(device)

    # 获取测试集
    testloader = get_testloader(batch_size=args.batch_size)

    # 评估
    acc = evaluate(model, testloader)
    print(f"Accuracy on CIFAR-10 test set: {acc:.2f}%")
