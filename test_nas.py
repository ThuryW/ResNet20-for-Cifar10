import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import json
from search_and_finetune import SearchableResNet, ChebyshevReLU, _weights_init, searchable_resnet20

# --- 1. Recreate the necessary model classes from your original script ---

# NOTE: You MUST copy/paste the definitions for Config, ChebyshevReLU, 
# ChannelWiseArchReLU, _weights_init, LambdaLayer, SearchableBasicBlock, 
# and SearchableResNet here.
# For simplicity, let's assume they are defined.

# We will need a way to build the final model.
# One approach is to modify the Searchable modules to be "fixed".

class FixedChannelWiseReLU(nn.Module):
    def __init__(self, num_channels, chebyshev_coeffs, domain_min, domain_max, choices):
        super().__init__()
        self.num_channels = num_channels
        self.relu = nn.ReLU()
        self.cheby_relu = ChebyshevReLU(chebyshev_coeffs, domain_min, domain_max)
        self.choices = choices # The tensor of 0s and 1s

    def forward(self, x):
        out = torch.empty_like(x)
        for i in range(self.num_channels):
            if self.choices[i] == 0:  # Chebyshev
                out[:, i, :, :] = self.cheby_relu(x[:, i, :, :])
            else:  # ReLU
                out[:, i, :, :] = self.relu(x[:, i, :, :])
        return out

# A new ResNet class that uses the fixed layers
class FinalResNet(SearchableResNet):
    def __init__(self, block, num_blocks, num_classes=10, chebyshev_coeffs_config=None, arch_choices=None):
        super().__init__(block, num_blocks, num_classes, chebyshev_coeffs_config)
        self.arch_choices = arch_choices
        self._replace_layers()
        self.apply(_weights_init)

    def _replace_layers(self):
        # This function would iterate through all modules and replace
        # ChannelWiseArchReLU with FixedChannelWiseReLU based on arch_choices
        # This is a complex part and would need to be implemented carefully.
        # A simpler approach is to rebuild the model from scratch.
        # For this example, let's just assume we can load the state dict directly
        # into the original SearchableResNet, as long as it's not in training mode.
        pass

# --- 2. Configuration and Data Loading (same as original script) ---

class Config:
    PROJECT_DIR = '/home/wangtianyu/relu_finetune'
    CHEBYSHEV_COEFFS_PATH = os.path.join(PROJECT_DIR, 'scripts/single_relu/relu_coeffs.json')
    FINAL_MODEL_SAVE_PATH = os.path.join(PROJECT_DIR, 'final_channel_wise_model.pth')
    BATCH_SIZE = 64
    DOMAIN_MIN = -10.0
    DOMAIN_MAX = 10.0

def load_test_data(cfg):
    print('==> Preparing test data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_root = os.path.join(cfg.PROJECT_DIR, 'data')
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return testloader

# --- Main script starts here ---

if __name__ == '__main__':
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Chebyshev coefficients for the model definition
    with open(cfg.CHEBYSHEV_COEFFS_PATH, 'r') as f:
        coeffs = json.load(f)

    chebyshev_config = {
        'chebyshev_coeffs': coeffs,
        'domain_min': cfg.DOMAIN_MIN,
        'domain_max': cfg.DOMAIN_MAX,
        'gumbel_tau': 1.0, # This value doesn't matter for the final model
    }

    # Load the saved model data
    print(f"Loading final model from {cfg.FINAL_MODEL_SAVE_PATH}...")
    saved_data = torch.load(cfg.FINAL_MODEL_SAVE_PATH, map_location=device)
    model_state_dict = saved_data['state_dict']
    arch_choices = saved_data['arch_choices']
    print("Model data loaded.")

    # Recreate the original model.
    # Note: Because the saved state_dict contains parameters for the
    # arch search (e.g., 'arch_params'), we can just load the state_dict
    # into the original searchable model, but we must set it to eval() mode.
    # A more robust solution is to build a new model from scratch based on arch_choices.
    model = searchable_resnet20(chebyshev_coeffs_config=chebyshev_config).to(device)
    model.load_state_dict(model_state_dict)

    # --- 5. Set up test data loader ---
    testloader = load_test_data(cfg)

    # --- 6. Run evaluation loop ---
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    print(f"\nTest Accuracy of the final model: {accuracy:.2f}%")