import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import time
from model.resnet import resnet20

# ==== 配置 ====
model_path = "./base_models/20_ckpt_92.23.pth"
pruned_model_path = "./base_models/20_ckpt_pruned_finetuned.pth"
target_sparsity = 0.8  # 总体剪枝率
finetune_epochs = 20
batch_size = 128
learning_rate = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 创建时间戳目录 ====
start_time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
save_dir = os.path.join("./usp_ckpt", start_time_str)
os.makedirs(save_dir, exist_ok=True)

log_path = os.path.join(save_dir, "prune_log.txt")
best_model_path = os.path.join(save_dir, "best.pth")
latest_model_path = os.path.join(save_dir, "latest.pth")

# ==== 加载模型 ====
model = resnet20().to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint # Handle different checkpoint formats

# Adjust state_dict keys for potential DataParallel prefix
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith('module.'):
        new_state_dict[k[7:]] = v # Remove 'module.' prefix
    else:
        new_state_dict[k] = v # Keep as is
model.load_state_dict(new_state_dict)

# checkpoint = torch.load(model_path, map_location=device)
# model.load_state_dict(checkpoint['net'])

# ==== 自定义权重重要性 ====
def get_weight_importance(weight_tensor):
    if weight_tensor.dim() == 4 and weight_tensor.size(2) == 3 and weight_tensor.size(3) == 3:
        pos_importance = torch.tensor([
            [1., 1., 1.],
            [2., 3., 2.],
            [1., 1., 1.]
        ], device=weight_tensor.device)
        pos_importance = pos_importance.unsqueeze(0).unsqueeze(0)
        pos_importance = pos_importance.expand_as(weight_tensor)
        return weight_tensor.abs() * pos_importance
    else:
        return weight_tensor.abs()

# ==== 收集权重重要性 ====
all_importances = []
all_weights = []
layer_names = []
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        importance = get_weight_importance(module.weight.data)
        all_importances.append(importance.flatten())
        all_weights.append((name, module.weight))
        layer_names.append(name)

all_importances = torch.cat(all_importances)
num_params_to_prune = int(target_sparsity * all_importances.numel())
threshold = torch.topk(all_importances, num_params_to_prune, largest=False).values.max()

# ==== 剪枝并记录log ====
log_lines = []
total_params = 0
remaining_params = 0
position_stats = torch.zeros(3, 3)  # 卷积核位置保留统计
position_counts = torch.zeros(3, 3)  # 卷积核位置总数

for name, weight in all_weights:
    importance = get_weight_importance(weight.data)
    mask = (importance >= threshold).float()
    weight.data *= mask

    layer_total = weight.numel()
    layer_remaining = mask.sum().item()
    layer_sparsity = 1 - layer_remaining / layer_total

    total_params += layer_total
    remaining_params += layer_remaining

    if weight.dim() == 4 and weight.size(2) == 3:
        # 统计不同位置的保留比例
        for i in range(3):
            for j in range(3):
                pos_mask = mask[:, :, i, j]
                position_stats[i, j] += pos_mask.sum().item()
                position_counts[i, j] += pos_mask.numel()

    log_lines.append(f"{name}: 剪枝率={layer_sparsity*100:.2f}%")

overall_sparsity = 1 - remaining_params / total_params
log_lines.append(f"总体剪枝率={overall_sparsity*100:.2f}%")
log_lines.append("卷积核不同位置的保留比例：")
for i in range(3):
    for j in range(3):
        ratio = position_stats[i, j] / position_counts[i, j] if position_counts[i, j] > 0 else 0
        log_lines.append(f"k{i*3+j+1}: {ratio*100:.2f}% 保留")

os.makedirs(os.path.dirname(log_path), exist_ok=True)
with open(log_path, "w") as f:
    f.write("\n".join(log_lines))
print(f"剪枝完成，log 已保存到 {log_path}")

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

# ==== Finetune ====
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

def test():
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

best_acc = 0.0
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
    acc = test()
    print(f"Epoch [{epoch+1}/{finetune_epochs}] Test Acc: {acc:.2f}%")

    # 保存 best
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), best_model_path)

# ==== 保存 latest ====
torch.save(model.state_dict(), latest_model_path)
print(f"Best 模型已保存到: {best_model_path}")
print(f"Latest 模型已保存到: {latest_model_path}")


