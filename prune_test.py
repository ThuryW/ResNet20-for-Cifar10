import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune # 导入 PyTorch 剪枝模块

import torchvision
import torchvision.transforms as transforms

from model.resnet import * # 假设 resnet20 在这里定义
from model.squeezeNet import * # 未使用，但保留
from utils import AvgMeter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_test_data(args):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    testset = torchvision.datasets.CIFAR10(root = './data', train = False, download=True, transform = transform_test) # Added download=True for convenience
    testloader = torch.utils.data.DataLoader(testset, batch_size = args.batch_size, shuffle = False, num_workers = 2)

    # 只取前227张图片的索引
    start_image = 1135
    subset_indices = list(range(start_image, start_image + 227))  # 选择227张图片

    # 创建Subset数据集
    subset_dataset = torch.utils.data.Subset(testset, subset_indices)

    # 创建一个新的 DataLoader 用于这一子集
    subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=32, shuffle=False, num_workers=2) # Added num_workers for subset_loader

    return testloader, subset_loader

# ==============================================================================
# 新增的剪枝函数
# ==============================================================================
def prune_model(model, pruning_amount):
    """
    对模型进行非结构化剪枝。
    使用 L1 非结构化剪枝方法，即根据权重绝对值大小进行剪枝。

    Args:
        model (nn.Module): 要剪枝的 PyTorch 模型。
        pruning_amount (float): 剪枝的比例（0到1之间），例如 0.5 表示剪枝50%的权重。
    """
    print(f"Applying unstructured pruning with amount: {pruning_amount * 100:.2f}%")

    # 遍历模型的所有模块（层）
    for name, module in model.named_modules():
        # 我们只对 Conv2d 和 Linear 层的权重进行剪枝
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # PyTorch 的 prune.random_unstructured 或 prune.l1_unstructured
            # 这里我们使用l1_unstructured，根据L1范数（绝对值）进行剪枝
            # 它会创建一个与权重形状相同的掩码，将剪枝掉的权重位置置为0
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            # prune.remove_pruning(module, 'weight') # 如果需要在剪枝后移除钩子，但我们这里是评估，所以暂时不需要

    # 计算剪枝后的实际稀疏度
    total_params = 0
    pruned_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            total_params += module.weight.nelement() # 计算总权重数量
            # 计算非零权重数量
            pruned_params += torch.sum(module.weight == 0)
            # 注意：prune.l1_unstructured会在模块中添加一个weight_orig和一个weight_mask
            # 实际使用的权重是 module.weight_orig * module.weight_mask
            # 所以更准确的稀疏度应该基于最终生效的权重
            # 如果我们不remove_pruning，module.weight 就是已经应用掩码后的张量
    
    # 更精确的稀疏度计算：
    total_elements = 0
    zero_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if hasattr(module, 'weight_mask'): # 检查是否被剪枝
                total_elements += module.weight_mask.nelement()
                zero_elements += torch.sum(module.weight_mask == 0)
            elif hasattr(module, 'weight'): # 如果没有被prune过，但有weight
                total_elements += module.weight.nelement()
                zero_elements += torch.sum(module.weight == 0)

    actual_sparsity = (zero_elements / total_elements) * 100 if total_elements > 0 else 0
    print(f"Model sparsity after pruning: {actual_sparsity:.2f}%")

    return model

# ==============================================================================


def test_process(args, test_loader):
    # 1. 实例化 ResNet 模型
    net = resnet20()
    net = net.to(device)

    # 2. 加载 checkpoint
    checkpoint = torch.load(args.path)
    
    # 3. 处理 state_dict 的键，移除 'module.' 前缀（如果存在）
    # 检查 checkpoint 的键是否以 'module.' 开头
    # 你的错误信息表明 checkpoint 的键是没有 'module.' 前缀的
    # 但是为了通用性，我们最好还是检查一下
    state_dict = checkpoint['net']
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v # 移除 'module.' 前缀
        else:
            new_state_dict[k] = v # 保持原样

    # 4. 加载处理后的 state_dict 到原始模型
    net.load_state_dict(new_state_dict)

    # 5. 如果需要，再将模型包装到 DataParallel
    if device == 'cuda':
        # 只有在加载完权重后才包装DataParallel
        # 这样模型的内部结构（keys）与加载的state_dict匹配
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True # 只有在DataParallel包装后才设置benchmark

    # best_epoch = checkpoint['epoch']
    # print('best epoch:', best_epoch)

    # ==========================================================================
    # 在这里应用剪枝
    # 注意：如果 net 已经被 DataParallel 包装，prune_model 函数会遍历其内部模块
    # 这是正确的行为，因为 prune.l1_unstructured 会作用于实际的权重张量
    # ==========================================================================
    if args.prune_amount > 0:
        net = prune_model(net, args.prune_amount)
    # ==========================================================================

    net.eval()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    for image_batch, gt_batch in tqdm(test_loader):
        image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
        gt_batch = gt_batch.long()
        with torch.no_grad():
            pred_batch = net(image_batch)
            loss = criterion(pred_batch, gt_batch)
        loss_meter.add(loss.item(), image_batch.size(0))
        acc = (pred_batch.argmax(dim=-1).long() == gt_batch).float().mean()
        acc_meter.add(acc.item(), image_batch.size(0))
    test_loss = loss_meter.avg()
    test_acc = acc_meter.avg()

    return test_loss, test_acc

def test_result(start_time, test_loss, test_acc):
    l_str = 'Elapsed time {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))
    print(l_str)
    l_str = '***** Test set result ***** loss: {:1.4f}, accuracy {:1.2f}%'.format(test_loss, test_acc * 100)
    print(l_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'PyTorch CIFAR10 Test with Pruning')
    parser.add_argument('--batch_size', default = 512, type = int, help = 'test batch size')
    parser.add_argument('--path', default = '/home/wangtianyu/my_resnet20/checkpoint/20_ckpt_92.23.pth', type = str, help = 'test model path')
    parser.add_argument('--prune_amount', default = 0.0, type = float, help = 'Amount of unstructured pruning (0.0 to 1.0)') # 新增剪枝参数
    args = parser.parse_args()

    test_loader, subset_loader = load_test_data(args)

    start_time = time.time()
    # test_loss, test_acc = test_process(args, subset_loader) # 如果需要测试子集，取消注释
    test_loss, test_acc = test_process(args, test_loader)
    
    test_result(start_time, test_loss, test_acc)