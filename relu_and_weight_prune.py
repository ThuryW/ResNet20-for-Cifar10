#!/usr/bin/env python3
"""
run_relu_and_weight_prune.py

流程：
1) 构造 student：去掉 conv1 的初始 ReLU；Stage1 去掉所有 ReLU；Stage2/3 按 CaseB 去掉 relu2（保留 relu1）
2) 记录初始精度（no finetune）
3) 使用知识蒸馏(KD)在 CIFAR-10 上 finetune student（teacher=原始 model）
   - 保存： ./usp_ckpt/<timestamp>/relu_prune/best.pth, latest.pth, relu_prune_log.txt
4) 在 KD 后的 student 上做全局非结构化权重剪枝（Conv2d & Linear），位置重要性偏好 k5>k4,k6>others
   - 记录每层剪枝率、总体剪枝率、k1..k9 的保留比例
   - 保存 mask 后模型（未 finetune）和日志到 ./usp_ckpt/<timestamp>/weight_prune/
5) 对剪枝后模型做 supervised finetune（CE），保存 best/latest 和日志 weight_finetune_log.txt
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from model.resnet import resnet20

# ----------------------
# Config (adjustable)
# ----------------------
base_model_path = "./base_models/20_ckpt_92.23.pth"
save_root = "./checkpoints/combine_ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# KD (for ReLU removal recovery)
kd_epochs = 20
kd_batch_size = 128
kd_lr = 0.01
kd_alpha = 0.5
kd_T = 4.0

# Weight pruning
target_sparsity = 0.8  # 全局目标稀疏率 (e.g., 0.8 => 80% weights zeroed)
prune_finetune_epochs = 20
prune_finetune_batch_size = 128
prune_finetune_lr = 0.01

num_workers = 4

# ----------------------
# Utilities
# ----------------------
def make_timestamped_run_dir(root):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    run_dir = os.path.join(root, ts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def load_checkpoint_to_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt['net'] if 'net' in ckpt else ckpt # Handle different checkpoint formats

    # Adjust state_dict keys for potential DataParallel prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # Remove 'module.' prefix
        else:
            new_state_dict[k] = v # Keep as is
    model.load_state_dict(new_state_dict)

    # model.load_state_dict(ckpt)
    return model

# ----------------------
# Data (CIFAR-10)
# ----------------------
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
trainloader_kd = torch.utils.data.DataLoader(trainset, batch_size=kd_batch_size, shuffle=True, num_workers=num_workers)

trainloader_prune_ft = torch.utils.data.DataLoader(trainset, batch_size=prune_finetune_batch_size, shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=num_workers)

# ----------------------
# Eval
# ----------------------
def evaluate(model):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    return 100.0 * correct / total

# ----------------------
# KD loss
# ----------------------
def kd_loss_fn(student_logits, teacher_logits, labels, alpha, T):
    ce = F.cross_entropy(student_logits, labels)
    # KLDiv: log_softmax(student/T) vs softmax(teacher/T)
    kd = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                  F.softmax(teacher_logits / T, dim=1),
                  reduction='batchmean') * (T * T)
    return alpha * ce + (1 - alpha) * kd

# ----------------------
# Model modification (ReLU pruning)
# ----------------------
def build_student_caseB_remove_relus():
    """
    Build student with:
      - remove conv1 initial relu (relu_initial)
      - Stage1: remove all relu1 and relu2
      - Stage2 & Stage3: remove relu2 in each block (CaseB)
    """
    student = resnet20()
    # load base weights first (we will use teacher from same ckpt)
    _ = load_checkpoint_to_model(student, base_model_path)

    # remove conv1 initial relu
    if hasattr(student, "relu_initial"):
        student.relu_initial = nn.Identity()

    # Stage1: remove all relus
    for block in student.layer1:
        if hasattr(block, "relu1"):
            block.relu1 = nn.Identity()
        if hasattr(block, "relu2"):
            block.relu2 = nn.Identity()

    # Stage2 & Stage3: remove relu2 in each block (CaseB)
    for stage in [student.layer2, student.layer3]:
        for block in stage:
            if hasattr(block, "relu2"):
                block.relu2 = nn.Identity()

    return student

# ----------------------
# KD finetune routine (student <- teacher)
# ----------------------
def run_kd_finetune(student, teacher, save_dir, epochs=kd_epochs, batch_loader=trainloader_kd,
                    lr=kd_lr, alpha=kd_alpha, T=kd_T):
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pth")
    latest_path = os.path.join(save_dir, "latest.pth")
    log_path = os.path.join(save_dir, "relu_prune_log.txt")

    # move to device
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()

    init_acc = evaluate(student)
    print(f"[KD] Initial accuracy (after ReLU removal, before KD finetune): {init_acc:.2f}%")

    optimizer = torch.optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs//2), gamma=0.1)

    best_acc = 0.0
    log_lines = [f"Initial Acc (no finetune): {init_acc:.2f}%"]
    for ep in range(epochs):
        student.train()
        running_loss = 0.0
        for inputs, targets in batch_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                teacher_logits = teacher(inputs)
            student_logits = student(inputs)
            loss = kd_loss_fn(student_logits, teacher_logits, targets, alpha, T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        acc = evaluate(student)
        avg_loss = running_loss / len(batch_loader.dataset)
        log_lines.append(f"Epoch [{ep+1}/{epochs}] loss={avg_loss:.4f} test_acc={acc:.2f}%")
        print(log_lines[-1])
        if acc > best_acc:
            best_acc = acc
            torch.save(student.state_dict(), best_path)

    # save latest
    torch.save(student.state_dict(), latest_path)
    log_lines.append(f"Best Acc after KD finetune: {best_acc:.2f}%")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"[KD] Done. best/latest saved in {save_dir}, log saved to {log_path}")
    return student  # return finetuned student

# ----------------------
# Weight importance & global non-structured pruning
# ----------------------
def compute_positional_importance(weight_tensor):
    """
    For 3x3 conv kernels, apply position importance:
      pos matrix:
         [[1,1,1],
          [2,3,2],
          [1,1,1]]
    Multiply by abs(weight) to get importance.
    For other tensors (e.g., Linear), importance = abs(weight).
    """
    if weight_tensor.dim() == 4 and weight_tensor.size(2) == 3 and weight_tensor.size(3) == 3:
        device = weight_tensor.device
        pos = torch.tensor([[1.,1.,1.],
                            [2.,3.,2.],
                            [1.,1.,1.]], device=device)
        pos = pos.unsqueeze(0).unsqueeze(0)  # shape 1x1x3x3
        pos = pos.expand_as(weight_tensor)
        return weight_tensor.abs() * pos
    else:
        return weight_tensor.abs()

def global_unstructured_prune(model, target_sparsity):
    """
    Perform global unstructured pruning on Conv2d and Linear weights.
    Returns:
      - pruned_model (in-place)
      - prune_log (list of lines describing per-layer sparsity and position stats)
      - stats dict containing overall_sparsity and position retention matrix for 3x3 kernels
    """
    # collect importances and references
    importances = []
    param_refs = []  # tuples (name, param)
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            imp = compute_positional_importance(w).flatten()
            importances.append(imp)
            param_refs.append((name, module.weight))

    if len(importances) == 0:
        raise RuntimeError("No conv/linear weights found to prune.")

    all_imp = torch.cat(importances)
    total_params = all_imp.numel()
    num_prune = int(target_sparsity * total_params)
    if num_prune <= 0:
        threshold = -1e9
    else:
        # smallest num_prune importances to prune -> threshold is maximum of those smallest
        vals, _ = torch.topk(all_imp, k=num_prune, largest=False)
        threshold = vals.max().item()

    # apply masks and collect layer stats
    prune_log_lines = []
    total = 0
    remaining = 0
    position_stats = torch.zeros(3,3)  # kept counts
    position_counts = torch.zeros(3,3)  # total counts

    for name, param in param_refs:
        w = param.data
        imp = compute_positional_importance(w)
        mask = (imp >= threshold).float()
        # apply mask in-place
        param.data.mul_(mask)

        layer_total = w.numel()
        layer_remaining = int(mask.sum().item())
        layer_sparsity = 1.0 - (layer_remaining / layer_total)
        prune_log_lines.append(f"{name}: sparsity={layer_sparsity*100:.2f}% ({layer_remaining}/{layer_total})")

        total += layer_total
        remaining += layer_remaining

        # collect position stats for 3x3 convs
        if w.dim() == 4 and w.size(2) == 3 and w.size(3) == 3:
            # mask shape [out, in, 3,3]
            for i in range(3):
                for j in range(3):
                    pos_mask = mask[:, :, i, j]
                    position_stats[i,j] += pos_mask.sum().item()
                    position_counts[i,j] += pos_mask.numel()

    overall_sparsity = 1.0 - (remaining / total)
    prune_log_lines.append(f"Overall sparsity: {overall_sparsity*100:.2f}% ({remaining}/{total})")
    prune_log_lines.append("Position retention rates (k1..k9):")
    # map k index to i,j
    for i in range(3):
        for j in range(3):
            kept = position_stats[i,j].item()
            cnt = position_counts[i,j].item()
            ratio = (kept / cnt * 100.0) if cnt > 0 else 0.0
            prune_log_lines.append(f"k{ i*3 + j + 1 }: kept {kept:.0f}/{cnt:.0f} => {ratio:.2f}%")

    stats = {
        "overall_sparsity": overall_sparsity,
        "position_stats": (position_stats, position_counts)
    }
    return model, prune_log_lines, stats

# ----------------------
# Supervised finetune (after weight pruning)
# ----------------------
def supervised_finetune(model, save_dir, epochs=prune_finetune_epochs, batch_loader=trainloader_prune_ft,
                        lr=prune_finetune_lr):
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pth")
    latest_path = os.path.join(save_dir, "latest.pth")
    log_path = os.path.join(save_dir, "weight_finetune_log.txt")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,epochs//2), gamma=0.1)

    best_acc = 0.0
    log_lines = [f"Start supervised finetune after weight pruning."]
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in batch_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        scheduler.step()

        acc = evaluate(model)
        avg_loss = running_loss / len(batch_loader.dataset)
        log_lines.append(f"Epoch [{ep+1}/{epochs}] loss={avg_loss:.4f} test_acc={acc:.2f}%")
        print(log_lines[-1])
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), best_path)

    # save latest
    torch.save(model.state_dict(), latest_path)
    log_lines.append(f"Best Acc after prune-fineturn: {best_acc:.2f}%")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))
    print(f"[Prune Finetune] Done. best/latest/log saved in {save_dir}")
    return model

# ----------------------
# Main orchestrator
# ----------------------
def main():
    run_dir = make_timestamped_run_dir(save_root)
    print("Run dir:", run_dir)

    # 1) Prepare teacher and student (ReLU pruning)
    teacher = resnet20()
    teacher = load_checkpoint_to_model(teacher, base_model_path)
    teacher.eval()

    student = build_student_caseB_remove_relus()  # ReLU removed per your spec

    relu_save_dir = os.path.join(run_dir, "relu_prune")
    os.makedirs(relu_save_dir, exist_ok=True)

    # 2) KD finetune
    student = run_kd_finetune(student, teacher, relu_save_dir,
                              epochs=kd_epochs, batch_loader=trainloader_kd,
                              lr=kd_lr, alpha=kd_alpha, T=kd_T)

    # 3) Weight pruning (global)
    # work on the KD-finetuned student
    prune_save_dir = os.path.join(run_dir, "weight_prune")
    os.makedirs(prune_save_dir, exist_ok=True)
    # make a copy of student before pruning for safety (state_dict)
    pre_prune_state = student.state_dict()

    # perform global unstructured prune
    student = student.to("cpu")  # move to cpu for manipulation if needed
    pruned_student, prune_log_lines, prune_stats = global_unstructured_prune(student, target_sparsity)

    # save the pruned-but-not-finetuned model
    pruned_model_path = os.path.join(prune_save_dir, "pruned_before_finetune.pth")
    torch.save(pruned_student.state_dict(), pruned_model_path)
    # write prune log
    prune_log_path = os.path.join(prune_save_dir, "weight_prune_log.txt")
    with open(prune_log_path, "w") as f:
        f.write("\n".join(prune_log_lines))
    print(f"[Prune] Completed global pruning. Log saved to {prune_log_path}. Model saved to {pruned_model_path}")

    # 4) Supervised finetune after pruning
    # move model back to device
    pruned_student = pruned_student.to(device)
    final_model = supervised_finetune(pruned_student, prune_save_dir,
                                      epochs=prune_finetune_epochs, batch_loader=trainloader_prune_ft, lr=prune_finetune_lr)

    # copy KD-stage logs into run dir as summary (optional)
    summary_path = os.path.join(run_dir, "summary.txt")
    with open(summary_path, "w") as s:
        s.write("Summary of run\n")
        s.write(f"Base checkpoint: {base_model_path}\n")
        s.write(f"Run time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n\n")
        # KD summary
        kd_log = os.path.join(relu_save_dir, "relu_prune_log.txt")
        if os.path.exists(kd_log):
            s.write("KD log (relu removal) snippet:\n")
            with open(kd_log, "r") as f:
                lines = f.readlines()
            s.writelines(lines[-10:])  # last 10 lines
            s.write("\n\n")
        # prune summary
        s.write("Prune log (weight pruning) snippet:\n")
        with open(prune_log_path, "r") as f:
            lines = f.readlines()
        s.writelines(lines[-20:])

    print("All done. Run directory:", run_dir)

if __name__ == "__main__":
    main()
