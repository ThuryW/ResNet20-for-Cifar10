import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
import os
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast

# --- Helpers and Configuration ---

class Config:
    PROJECT_DIR = '/home/wangtianyu/relu_finetune'
    CHEBYSHEV_COEFFS_PATH = os.path.join(PROJECT_DIR, 'scripts/single_relu/relu_coeffs.json')
    PRETRAINED_MODEL_PATH = os.path.join(PROJECT_DIR, 'base_models/20_ckpt_92.23.pth')
    FINAL_MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'final_channel_wise_model.pth')
    
    NUM_EPOCHS = 10
    BATCH_SIZE = 64
    LEARNING_RATE_WEIGHTS = 1e-3
    LEARNING_RATE_ARCH = 3e-4
    WEIGHT_DECAY = 1e-4
    GUMBEL_TAU = 1.0
    
    DOMAIN_MIN = -10.0
    DOMAIN_MAX = 10.0

# ... [load_data 函数无变化] ...

def load_data(cfg):
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_root = os.path.join(cfg.PROJECT_DIR, 'data')
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return trainloader, testloader
    
class ChebyshevReLU(nn.Module):
    """
    CHANGED: 在计算前加入了 torch.clamp 来防止数值爆炸。
    """
    def __init__(self, coeffs, domain_min=-10.0, domain_max=10.0):
        super().__init__()
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.register_buffer('coeffs', torch.tensor(coeffs, dtype=torch.float32))

    def forward(self, x):
        # KEY FIX: 将输入限制在拟合时的安全域内
        x_clamped = torch.clamp(x, self.domain_min, self.domain_max)

        # 使用被截断后的值进行后续计算
        x_norm = (2 * x_clamped - (self.domain_max + self.domain_min)) / (self.domain_max - self.domain_min)
        
        T0 = torch.ones_like(x_norm)
        if len(self.coeffs) == 1:
            return self.coeffs[0] * T0
        T1 = x_norm
        out = self.coeffs[0] * T0 + self.coeffs[1] * T1
        
        for i in range(2, len(self.coeffs)):
            Tn = 2 * x_norm * T1 - T0
            out += self.coeffs[i] * Tn
            T0, T1 = T1, Tn
            
        return out

# ... [ChannelWiseArchReLU 和 ResNet 模型定义无变化] ...
class ChannelWiseArchReLU(nn.Module):
    def __init__(self, num_channels, chebyshev_coeffs, domain_min, domain_max, gumbel_tau=1.0):
        super().__init__()
        self.num_channels = num_channels
        self.gumbel_tau = gumbel_tau
        self.relu = nn.ReLU()
        self.cheby_relu = ChebyshevReLU(chebyshev_coeffs, domain_min, domain_max)
        self.arch_params = nn.Parameter(1e-3 * torch.randn(num_channels, 2))

    def forward(self, x):
        gumbel_weights = F.gumbel_softmax(self.arch_params, tau=self.gumbel_tau, hard=False)
        shape_for_broadcast = (2, 1, self.num_channels, 1, 1)
        channel_weights = gumbel_weights.T.view(shape_for_broadcast)
        out_cheby = self.cheby_relu(x)
        out_relu = self.relu(x)
        output = channel_weights[0] * out_cheby + channel_weights[1] * out_relu
        return output

    def get_final_choice(self):
        return torch.argmax(self.arch_params, dim=1)

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class SearchableBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A', chebyshev_coeffs_config=None):
        super(SearchableBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = ChannelWiseArchReLU(planes, **chebyshev_coeffs_config)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = ChannelWiseArchReLU(planes, **chebyshev_coeffs_config)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        pre_x = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(pre_x)
        out = self.relu2(out)
        return out

class SearchableResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, chebyshev_coeffs_config=None):
        super(SearchableResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu_initial = ChannelWiseArchReLU(16, **chebyshev_coeffs_config)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, chebyshev_coeffs_config=chebyshev_coeffs_config)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, chebyshev_coeffs_config=chebyshev_coeffs_config)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, chebyshev_coeffs_config=chebyshev_coeffs_config)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, chebyshev_coeffs_config):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, chebyshev_coeffs_config=chebyshev_coeffs_config))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu_initial(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_arch_parameters(self):
        for name, param in self.named_parameters():
            if 'arch_params' in name:
                yield param
    
    def get_weight_parameters(self):
        for name, param in self.named_parameters():
            if 'arch_params' not in name:
                yield param
    
    def print_arch_choices(self):
        print("\n--- Final Architecture Choices (0=Chebyshev, 1=ReLU) ---")
        for name, module in self.named_modules():
            if isinstance(module, ChannelWiseArchReLU):
                choices = module.get_final_choice().cpu().numpy()
                cheby_count = np.sum(choices == 0)
                relu_count = np.sum(choices == 1)
                print(f"Layer '{name}':")
                print(f"  - Chebyshev replacements: {cheby_count}/{module.num_channels}")
                print(f"  - ReLU kept: {relu_count}/{module.num_channels}")
        print("----------------------------------------------------------\n")
        
def searchable_resnet20(chebyshev_coeffs_config):
    return SearchableResNet(SearchableBasicBlock, [3, 3, 3], chebyshev_coeffs_config=chebyshev_coeffs_config)

if __name__ == '__main__':
    # ... [初始化代码无变化] ...
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    with open(cfg.CHEBYSHEV_COEFFS_PATH, 'r') as f:
        coeffs = json.load(f)
    
    chebyshev_config = {
        'chebyshev_coeffs': coeffs,
        'domain_min': cfg.DOMAIN_MIN,
        'domain_max': cfg.DOMAIN_MAX,
        'gumbel_tau': cfg.GUMBEL_TAU,
    }

    model = searchable_resnet20(chebyshev_coeffs_config=chebyshev_config).to(device)

    if os.path.exists(cfg.PRETRAINED_MODEL_PATH):
        print(f"Loading pre-trained weights from {cfg.PRETRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(cfg.PRETRAINED_MODEL_PATH, map_location=device), strict=False)

    optimizer_weights = torch.optim.Adam(model.get_weight_parameters(), lr=cfg.LEARNING_RATE_WEIGHTS, weight_decay=cfg.WEIGHT_DECAY)
    optimizer_arch = torch.optim.Adam(model.get_arch_parameters(), lr=cfg.LEARNING_RATE_ARCH, betas=(0.5, 0.999), weight_decay=0)
    criterion = nn.CrossEntropyLoss()

    trainloader, testloader = load_data(cfg)
    arch_update_loader = testloader 
    weight_update_loader = trainloader
    
    print(f"Starting NAS search and fine-tuning for {cfg.NUM_EPOCHS} epochs...")

    # NEW: 添加异常检测上下文管理器
    with torch.autograd.detect_anomaly():
        for epoch in range(cfg.NUM_EPOCHS):
            model.train()
            print(f"\n--- Epoch {epoch+1}/{cfg.NUM_EPOCHS} ---")
            
            # a. 训练架构
            for i, (inputs, targets) in enumerate(arch_update_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_arch.zero_grad()
                
                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer_arch)
                scaler.update()

                if i % 20 == 0:
                    print(f"  [Arch Step {i}] Arch Loss: {loss.item():.4f}")

            # b. 训练权重
            for i, (inputs, targets) in enumerate(weight_update_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer_weights.zero_grad()

                with autocast(enabled=(device.type == 'cuda')):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer_weights)
                scaler.update()

                if i % 50 == 0:
                     print(f"  [Weight Step {i}] Weight Loss: {loss.item():.4f}")

    # ... [结束和保存部分无变化] ...
    print("\nNAS Search and fine-tuning finished.")
    model.print_arch_choices()
    print("Creating and saving the final model with discrete architecture...")
    final_model_state = {
        'state_dict': model.state_dict(),
        'arch_choices': {name: module.get_final_choice() 
                         for name, module in model.named_modules() 
                         if isinstance(module, ChannelWiseArchReLU)}
    }
    torch.save(final_model_state, cfg.FINAL_MODEL_SAVE_PATH)
    print(f"Final model and architecture choices saved to: {cfg.FINAL_MODEL_SAVE_PATH}")