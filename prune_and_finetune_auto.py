import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from datetime import datetime
import argparse
import logging # Import logging module
import sys     # Import sys for stdout

# IMPORTANT: Directly import resnet20 from your model directory.
# Ensure your 'model' directory is in the Python path or current working directory.
try:
    from model.resnet import resnet20
except ImportError:
    print("Error: Could not import resnet20 from model/resnet.py.")
    print("Please ensure you have a 'model' directory in the same location as this script,")
    print("and that 'resnet.py' exists within it and defines a 'resnet20' function.")
    exit(1) # Exit if the required model cannot be imported

# ==============================================================================
# Device Configuration
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AvgMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

# ==============================================================================
# Data Loading Function (Integrated)
# ==============================================================================
def load_data(args):
    """Loads CIFAR-10 training and test datasets."""
    logging.info('==> Preparing data..')
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader

# ==============================================================================
# Training Function (Integrated)
# ==============================================================================
def train_model(model, train_loader, args, test_loader_for_eval, epoch_start_val=0, save_dir=None):
    """Trains the model for specified epochs."""
    logging.info(f"\n--- Starting training for {args.epochs} total epochs ---")
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=(args.epochs//3), gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    
    # If resuming, load optimizer and scheduler states
    if args.resume and args.resume_path:
        logging.info(f"Loading optimizer and scheduler states from {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=device)
        if 'optimizer' in checkpoint and 'scheduler' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            best_acc = checkpoint.get('acc', 0.0)
            epoch_start_val = checkpoint.get('epoch', 0) + 1 # Start from next epoch
            logging.info(f"Resumed optimizer, scheduler, and best_acc: {best_acc * 100:.2f}% from epoch {epoch_start_val-1}")
        else:
            logging.info("Optimizer/Scheduler state not found in checkpoint. Starting fresh.")


    for epoch in range(epoch_start_val, args.epochs):
        logging.info(f'\nTrain Epoch: {epoch+1}/{args.epochs} | LR: {scheduler.get_last_lr()[0]:.6f}')
        loss_meter = AvgMeter()
        acc_meter = AvgMeter()

        with tqdm(total=len(train_loader), desc=f"Train Epoch {epoch+1}", file=sys.stdout) as pbar: # tqdm output to sys.stdout
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

                pbar.set_postfix({'loss': loss_meter.avg, 'acc': f'{acc_meter.avg * 100:.2f}%'})
                pbar.update(1)

        scheduler.step()
        
        # Evaluate after each epoch
        current_loss, current_acc = test_model(model, test_loader_for_eval, f"Epoch {epoch+1} Test")
        
        # Save best model and latest checkpoint
        is_best = current_acc > best_acc
        if is_best:
            best_acc = current_acc
            logging.info(f"New best accuracy: {best_acc * 100:.2f}%. Saving best model...")
            # Ensure the directory exists before saving
            os.makedirs(save_dir, exist_ok=True)
            # When saving, save the state_dict which still includes _orig and _mask
            torch.save(model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(), 
                       os.path.join(save_dir, 'best_model.pth'))

        logging.info("Saving latest checkpoint for resuming...")
        # Ensure the directory exists before saving
        os.makedirs(save_dir, exist_ok=True)
        state = {
            'net': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'acc': current_acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(state, os.path.join(save_dir, 'latest_checkpoint.pth'))

    logging.info("--- Training complete ---")
    return model

# ==============================================================================
# Test Function (Integrated)
# ==============================================================================
def test_model(model, test_loader, description="Test"):
    """Evaluates model performance."""
    model.eval()
    loss_meter = AvgMeter()
    acc_meter = AvgMeter()
    criterion = nn.CrossEntropyLoss()

    logging.info(f"\n--- Starting {description} ---")
    with torch.no_grad():
        for image_batch, gt_batch in tqdm(test_loader, desc=description, file=sys.stdout): # tqdm output to sys.stdout
            image_batch, gt_batch = image_batch.to(device), gt_batch.to(device)
            pred_batch = model(image_batch)
            loss = criterion(pred_batch, gt_batch.long())
            loss_meter.add(loss.item(), image_batch.size(0))
            acc = (pred_batch.argmax(dim=-1).long() == gt_batch).float().mean()
            acc_meter.add(acc.item(), image_batch.size(0))

    test_loss = loss_meter.avg
    test_acc = acc_meter.avg

    logging.info(f"--- {description} Result --- Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    return test_loss, test_acc

# ==============================================================================
# 通道剪枝（结构化 L1 norm）
# ==============================================================================
def structured_prune_conv_layers(model, amount=0.5):
    logging.info(f"\n==> Applying structured channel pruning (L1 norm) with amount={amount}")
    if amount == 0:
        logging.info("  Skipping structured pruning as amount is 0.")
        return model

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # L1 norm on output channels (dim=0)
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
            # DO NOT CALL prune.remove() here to keep the pruned weights fixed at zero during fine-tuning.

    logging.info("  Structured pruning applied. Performing internal sparsity check...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Access the pruned weight through the reparametrization
            weight = getattr(module, 'weight') # This will be the masked weight
            num_zeros = torch.sum(weight == 0).item()
            total_elements = weight.numel()
            if total_elements > 0:
                sparsity_percentage = (num_zeros / total_elements) * 100
                logging.info(f"    Layer: {name}, Structured Weight Sparsity: {sparsity_percentage:.2f}% ({num_zeros}/{total_elements})")
    return model


# ==============================================================================
# 非结构化剪枝 (L1 norm)
# ==============================================================================
def unstructured_prune_weights(model, amount=0.5):
    logging.info(f"\n==> Applying unstructured weight pruning (L1 norm) with amount={amount}")
    if amount == 0:
        logging.info("  Skipping unstructured pruning as amount is 0.")
        return model

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # DO NOT CALL prune.remove() here to keep the pruned weights fixed at zero during fine-tuning.

    logging.info("  Unstructured pruning applied. Performing internal sparsity check...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Access the pruned weight through the reparametrization
            weight = getattr(module, 'weight') # This will be the masked weight
            num_zeros = torch.sum(weight == 0).item()
            total_elements = weight.numel()
            if total_elements > 0:
                sparsity_percentage = (num_zeros / total_elements) * 100
                logging.info(f"    Layer: {name}, Unstructured Weight Sparsity: {sparsity_percentage:.2f}% ({num_zeros}/{total_elements})")
    return model


# ==============================================================================
# 剪枝率计算 (更详细) - 适用于任何状态的模型
# ==============================================================================
def compute_total_sparsity(model):
    total_prunable_elements = 0 # sum of elements in weights of Conv2d/Linear layers
    zero_prunable_elements = 0  # sum of zero elements in weights of Conv2d/Linear layers
    
    logging.info("\n==> Layer-wise sparsity:")
    for name, module in model.named_modules():
        # Check if the module is a type that can be pruned (Conv2d or Linear)
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight') and module.weight is not None:
                # After prune.remove(), module.weight is the final, masked weight.
                # If prune.remove() was NOT called, module.weight is the original weight.
                current_weight = module.weight
                
                current_total_elements = current_weight.numel()
                current_zero_elements = torch.sum(current_weight == 0).item()

                total_prunable_elements += current_total_elements
                zero_prunable_elements += current_zero_elements

                layer_sparsity = (current_zero_elements / current_total_elements) * 100 if current_total_elements > 0 else 0
                logging.info(f"  {name}.weight: {layer_sparsity:.2f}% zeroed")
            
            # Bias parameters are generally not pruned, but contribute to overall sparsity.
            # If you want to log bias sparsity per layer, uncomment below:
            # if hasattr(module, 'bias') and module.bias is not None:
            #     bias_num_zeros = torch.sum(module.bias == 0).item()
            #     bias_total_elements = module.bias.numel()
            #     bias_sparsity = (bias_num_zeros / bias_total_elements) * 100 if bias_total_elements > 0 else 0
            #     logging.info(f"  {name}.bias: {bias_sparsity:.2f}% zeroed (Bias)")
        
    if total_prunable_elements > 0:
        overall_prunable_sparsity = zero_prunable_elements / total_prunable_elements
        logging.info(f"\n==> Overall sparsity for PRUNABLE weights (Conv2d & Linear): {overall_prunable_sparsity * 100:.2f}%")
    else:
        logging.info("\nNo prunable weights (Conv2d & Linear) found to calculate specific sparsity, or they have 0 elements.")

    # Calculate total model sparsity including ALL parameters (weights, biases, batchnorm, etc.)
    total_all_params = 0
    zero_all_params = 0
    for param in model.parameters():
        total_all_params += param.numel()
        zero_all_params += torch.sum(param == 0).item()
        
    overall_model_sparsity = zero_all_params / total_all_params if total_all_params > 0 else 0
    logging.info(f"==> Total model sparsity (including ALL parameters): {overall_model_sparsity * 100:.2f}%")
    return overall_model_sparsity


# ==============================================================================
# 主流程
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning and Fine-tuning')
    parser.add_argument('--batch_size', default=512, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs to fine-tune')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume_path', default=None, type=str, help='path to latest checkpoint (if resume is true)')
    parser.add_argument('--pretrained_model_path', default='/home/wangtianyu/my_resnet20/base_models/20_ckpt_92.23.pth', type=str, help='path to the pretrained model checkpoint')
    parser.add_argument('--structured_pruning_amount', default=0.0, type=float, help='amount for structured pruning (0.0 means no structured pruning)')
    parser.add_argument('--unstructured_pruning_amount', default=0.9, type=float, help='amount for unstructured pruning (e.g., 0.9 for 90%)')
    parser.add_argument('--min_acc_drop', default=0.01, type=float, help='maximum allowed accuracy drop after pruning and finetuning')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save fine-tuned model checkpoints and logs')
    # Removed default for save_final_pruned_model_path here, will set dynamically
    parser.add_argument('--save_final_pruned_model_path', type=str, default=None, 
                        help='Optional: Path to save the final fine-tuned model after permanently applying pruning (removing _orig/_mask)')

    args = parser.parse_args()

    # Create the base output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a timestamped subdirectory within the output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    finetuned_save_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(finetuned_save_dir, exist_ok=True)

    # --- Setup Logging ---
    log_file_path = os.path.join(finetuned_save_dir, 'training_log.txt')
    # Basic configuration for logging: INFO level, output to console and file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout), # Output to console
            logging.FileHandler(log_file_path) # Output to a file
        ]
    )
    logging.info(f"All outputs (checkpoints, logs) will be saved in: {finetuned_save_dir}")
    # --- End Setup Logging ---

    # --- Set default for save_final_pruned_model_path ---
    if args.save_final_pruned_model_path is None:
        args.save_final_pruned_model_path = os.path.join(finetuned_save_dir, 'final_solidified_model.pth')
        logging.info(f"Defaulting save_final_pruned_model_path to: {args.save_final_pruned_model_path}")
    # --- End set default ---

    # 1. Load Data
    train_loader, test_loader = load_data(args)

    # 2. Load Pretrained ResNet20
    model = resnet20().to(device)
    
    logging.info(f"\nLoading pretrained model from: {args.pretrained_model_path}")
    try:
        checkpoint = torch.load(args.pretrained_model_path, map_location=device)
        state_dict = checkpoint['net'] if 'net' in checkpoint else checkpoint # Handle different checkpoint formats
        
        # Adjust state_dict keys for potential DataParallel prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # Remove 'module.' prefix
            else:
                new_state_dict[k] = v # Keep as is
        model.load_state_dict(new_state_dict)
        logging.info("Pretrained model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading pretrained model from {args.pretrained_model_path}: {e}")
        logging.error("Please ensure the path is correct and the file is a valid PyTorch checkpoint.")
        exit() # Exit if pretrained model cannot be loaded

    model.eval()

    # 3. Evaluate original accuracy
    _, base_acc = test_model(model, test_loader, description="Original Model")

    # Ensure the model is unwrapped from DataParallel before pruning
    if isinstance(model, nn.DataParallel):
        logging.info("Unwrapping model from DataParallel for pruning...")
        model = model.module

    # 4. Apply Structured Pruning
    # IMPORTANT: Pruning functions DO NOT call prune.remove() here to maintain pruning during fine-tuning.
    model = structured_prune_conv_layers(model, amount=args.structured_pruning_amount)

    # 5. Apply Unstructured Pruning
    # IMPORTANT: Pruning functions DO NOT call prune.remove() here to maintain pruning during fine-tuning.
    model = unstructured_prune_weights(model, amount=args.unstructured_pruning_amount)

    # 6. Verify that the pruning masks are active after application
    # This check is just for confirmation *before* fine-tuning.
    logging.info("\nVerifying pruning masks after application (before fine-tuning):")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                logging.info(f"  Module {name} is pruned. Active weight sparsity: {(torch.sum(module.weight == 0).item() / module.weight.numel()) * 100:.2f}%")
            # else: # Uncomment to see modules not directly pruned by `torch.nn.utils.prune`
            #     logging.info(f"  Module {name} is NOT pruned by `torch.nn.utils.prune` mechanism (no mask).")
    
    # 7. Fine-tune the pruned model
    # Wrap the model in DataParallel again for training if using CUDA
    if device == 'cuda':
        logging.info("\nWrapping model in DataParallel for fine-tuning...")
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    finetuned_model = train_model(model, train_loader, args, test_loader, save_dir=finetuned_save_dir)

    # 8. Evaluate finetuned model (load best checkpoint)
    # Create a fresh model instance to load the best fine-tuned state_dict
    # NOTE: This `final_model` will initially still have _orig and _mask if the saved checkpoint had them.
    final_model = resnet20().to(device)
    
    # Load the state dict from the best_model.pth in the timestamped directory
    finetuned_checkpoint_path = os.path.join(finetuned_save_dir, 'best_model.pth')
    state_dict_finetuned = None

    if os.path.exists(finetuned_checkpoint_path):
        logging.info(f"\nLoading best_model.pth from {finetuned_checkpoint_path} for final evaluation.")
        state_dict_finetuned = torch.load(finetuned_checkpoint_path, map_location=device)
    else:
        logging.warning(f"\nWarning: best_model.pth not found in {finetuned_save_dir}. Attempting to load latest_checkpoint.pth instead.")
        finetuned_checkpoint_path = os.path.join(finetuned_save_dir, 'latest_checkpoint.pth')
        if os.path.exists(finetuned_checkpoint_path):
            logging.info("Loading latest_checkpoint.pth for final evaluation.")
            checkpoint_data = torch.load(finetuned_checkpoint_path, map_location=device)
            state_dict_finetuned = checkpoint_data['net'] # Assuming 'net' key holds the model state
        else:
            raise FileNotFoundError(f"Neither best_model.pth nor latest_checkpoint.pth found in {finetuned_save_dir}")

    # Adjust state_dict keys for potential DataParallel prefix
    new_state_dict_finetuned = {}
    for k, v in state_dict_finetuned.items():
        if k.startswith('module.'):
            new_state_dict_finetuned[k[7:]] = v # Remove 'module.' prefix
        else:
            new_state_dict_finetuned[k] = v # Keep as is
    
    # Apply dummy pruning to final_model before loading state_dict to handle _orig/_mask
    logging.info("Applying dummy pruning to final_model to prepare for loading state_dict...")
    for name, module in final_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=0.01)
    logging.info("Dummy pruning applied to final_model.")

    final_model.load_state_dict(new_state_dict_finetuned)
    logging.info("Fine-tuned model state_dict loaded successfully into final_model.")


    # --- NEW STEP: Permanently apply pruning to final_model ---
    logging.info("\n--- Permanently applying pruning to final_model (prune.remove()) ---")
    for name, module in final_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if prune.is_pruned(module):
                prune.remove(module, 'weight')
                # logging.info(f"  Removed pruning reparametrization for {name}.weight") # Uncomment for detailed log
    logging.info("Pruning permanently applied to final_model.")
    # --- END NEW STEP ---

    # Optional: Save the final model after integrating pruning
    if args.save_final_pruned_model_path:
        os.makedirs(os.path.dirname(args.save_final_pruned_model_path), exist_ok=True)
        # Save the state_dict of the model AFTER prune.remove()
        torch.save(final_model.state_dict(), args.save_final_pruned_model_path)
        logging.info(f"\nFinal pruned model (weights solidified) saved to: {args.save_final_pruned_model_path}")

    # Verify sparsity of the loaded and now solidified `final_model` before evaluation
    logging.info("\nVerifying sparsity of the loaded and solidified final model:")
    compute_total_sparsity(final_model) # This compute_total_sparsity function is compatible with solidified weights
    
    # Wrap the final_model in DataParallel for evaluation if using CUDA
    if device == 'cuda':
        final_model = torch.nn.DataParallel(final_model)
        cudnn.benchmark = True

    _, final_acc = test_model(final_model, test_loader, description="Pruned & Finetuned Model (Solidified)")

    # 9. Accuracy Check
    acc_drop = base_acc - final_acc
    logging.info(f"\n==> Original Accuracy: {base_acc * 100:.2f}%")
    logging.info(f"==> Pruned & Finetuned (Solidified) Accuracy: {final_acc * 100:.2f}%")
    logging.info(f"==> Absolute Accuracy Drop: {acc_drop * 100:.2f}%")
    
    # Optional assert based on your previous code, uncomment if needed
    # assert acc_drop < args.min_acc_drop, f"Accuracy drop ({acc_drop * 100:.2f}%) is too high! (Allowed: <{args.min_acc_drop * 100:.2f}%)"

    logging.info("\nPruning + Fine-tuning Complete!")


if __name__ == '__main__':
    main()