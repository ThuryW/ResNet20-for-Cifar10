import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import json
import numpy as np
import os
import sys
import random

# 假设 resnet.py 和该脚本在同一目录下
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from resnet import resnet20, BasicBlock

# ==============================================================================
# Step 1: Custom ReLU Layer Implementation (Channel-wise)
# ==============================================================================
class ChannelWiseCustomReLU(nn.Module):
    def __init__(self, num_channels, json_path, domain_min=-10.0, domain_max=10.0):
        super(ChannelWiseCustomReLU, self).__init__()
        self.num_channels = num_channels
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.relu = nn.ReLU(inplace=True)
        
        # Load Chebyshev coefficients
        self.coeffs = self._load_coefficients(json_path)

        # The mask is the core of the channel-wise search.
        # It's a learnable parameter that decides which channels use custom ReLU.
        # 0 for standard ReLU, 1 for custom ReLU.
        # We use a tensor here, but a real NAS might use a more complex
        # search-space representation (e.g., categorical choices).
        self.mask = nn.Parameter(torch.zeros(num_channels), requires_grad=False)
        
    def _load_coefficients(self, json_path):
        """Loads coefficients from a JSON file."""
        with open(json_path, 'r') as f:
            coeffs_list = json.load(f)
        return torch.tensor(coeffs_list, dtype=torch.float32)

    def _chebval(self, x, c):
        """
        Evaluate a Chebyshev series at points x using Clenshaw algorithm.
        """
        c = c.to(x.device)
        x2 = 2 * x
        
        if len(c) == 1:
            return c[0] + torch.zeros_like(x)
        elif len(c) == 2:
            return c[0] + c[1] * x

        c0 = c[-2]
        c1 = c[-1]
        
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1 * x2
            c1 = tmp
            
        return c0 + c1 * x
        
    def forward(self, x):
        # Calculate custom ReLU output
        x_normalized = (2 * x - (self.domain_min + self.domain_max)) / (self.domain_max - self.domain_min)
        custom_relu_out = self._chebval(x_normalized, self.coeffs)
        
        # Calculate standard ReLU output
        standard_relu_out = self.relu(x)

        # Apply the mask to combine the outputs
        # The mask will be a 1D tensor of shape (num_channels)
        # We need to reshape it to match the input tensor (batch, channels, height, width)
        mask_reshaped = self.mask.view(1, self.num_channels, 1, 1).to(x.device)

        # Combine based on the mask:
        # result = mask * custom_relu_out + (1 - mask) * standard_relu_out
        out = mask_reshaped * custom_relu_out + (1 - mask_reshaped) * standard_relu_out
        
        return out


# ==============================================================================
# Step 2: Dynamic Layer Replacement
# ==============================================================================
def replace_relu_with_channel_wise_custom(model, json_path):
    """
    Dynamically replaces all nn.ReLU modules in the model with ChannelWiseCustomReLU.
    The number of channels is inferred from the preceding layer.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            # Find the parent module to infer the number of channels
            parent_name = name.rsplit('.', 1)[0]
            parent_module = model
            if parent_name:
                for sub_name in parent_name.split('.'):
                    parent_module = getattr(parent_module, sub_name)
            
            # Infer the number of output channels from the previous layer
            num_channels = None
            if isinstance(parent_module, nn.Conv2d):
                num_channels = parent_module.out_channels
            elif isinstance(parent_module, BasicBlock):
                # For BasicBlock, the second ReLU uses the output of the second conv
                if name.endswith('relu2'):
                    num_channels = parent_module.conv2.out_channels
                else: # The first ReLU uses the output of the first conv
                    num_channels = parent_module.conv1.out_channels
            elif name == 'relu_initial':
                 # The initial ReLU follows the first Conv2d
                 num_channels = model.conv1.out_channels

            if num_channels is None:
                raise ValueError(f"Could not infer number of channels for ReLU at '{name}'")

            new_relu_layer = ChannelWiseCustomReLU(num_channels, json_path)
            setattr(parent_module, name.split('.')[-1], new_relu_layer)
            print(f"Replaced ReLU at '{name}' with ChannelWiseCustomReLU (channels: {num_channels}).")
            
    return model

# ==============================================================================
# Step 3: Main NAS Search Loop (Framework)
# ==============================================================================
def train_and_eval(model, device, train_loader, test_loader, epochs=5, lr=0.1):
    """
    A simplified training and evaluation loop.
    Returns the final test accuracy.
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    accuracy = 100. * correct / len(test_loader.dataset)
    return accuracy

if __name__ == '__main__':
    # --- Configuration ---
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    chebyshev_json_path = '/home/wangtianyu/relu_finetune/scripts/single_relu/relu_coeffs.json'
    if not os.path.exists(chebyshev_json_path):
        print(f"Error: JSON file '{chebyshev_json_path}' not found.")
        print("Please ensure the file exists and contains Chebyshev coefficients.")
        sys.exit(1)

    # Data Loading
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # --- NAS Search Loop Placeholder ---
    # Here, you would choose and implement your NAS algorithm.
    # The following is a simple "random search" as a concrete example.
    
    num_nas_iterations = 5 # Number of architectures to sample and evaluate
    best_accuracy = 0.0
    best_config = None

    # Get a list of all ChannelWiseCustomReLU modules in the model for the search space
    # The search space is defined by the masks of these layers.
    dummy_model = resnet20()
    dummy_model = replace_relu_with_channel_wise_custom(dummy_model, chebyshev_json_path)
    all_channel_wise_relus = [m for m in dummy_model.modules() if isinstance(m, ChannelWiseCustomReLU)]

    print(f"\nStarting NAS search for {num_nas_iterations} iterations...")
    for i in range(num_nas_iterations):
        print(f"\n--- NAS Iteration {i+1}/{num_nas_iterations} ---")
        
        # 1. Generate a new candidate architecture (a new mask configuration)
        # This is where your NAS algorithm logic goes.
        # Here, we do a simple random search: for each mask, randomly set 50% of channels to use the custom ReLU.
        
        # Create a new model instance for each iteration
        model = resnet20().to(device)
        model = replace_relu_with_channel_wise_custom(model, chebyshev_json_path)
        
        # Apply the random mask to the newly created model
        for module in model.modules():
            if isinstance(module, ChannelWiseCustomReLU):
                # Randomly set half of the channels to be 'custom'
                random_mask = torch.zeros(module.num_channels, device=device)
                indices = torch.randperm(module.num_channels, device=device)[:module.num_channels // 2]
                random_mask[indices] = 1
                module.mask.data.copy_(random_mask)
        
        # 2. Train and evaluate the candidate model
        print("Training candidate architecture...")
        accuracy = train_and_eval(model, device, trainloader, testloader, epochs=5)
        
        print(f"Candidate accuracy: {accuracy:.2f}%")

        # 3. Update the best configuration
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_config = {name: module.mask.data.cpu().numpy().tolist() 
                           for name, module in model.named_modules() if isinstance(module, ChannelWiseCustomReLU)}
            print(f"New best accuracy found: {best_accuracy:.2f}%")
            
    print("\n" + "="*50)
    print("NAS search completed.")
    print(f"Best accuracy found: {best_accuracy:.2f}%")
    
    if best_config:
        print("Best configuration (masks):")
        # Save the best masks to a file
        with open('best_nas_config.json', 'w') as f:
            json.dump(best_config, f, indent=4)
        print("Best configuration saved to 'best_nas_config.json'.")