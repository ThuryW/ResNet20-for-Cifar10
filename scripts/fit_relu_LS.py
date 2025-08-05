import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
import argparse
import torchvision
import torchvision.transforms as transforms
from resnet import resnet20 # Make sure resnet.py is in your path or correctly imported
import logging
import json

def relu(x):
    """ReLU activation function for numpy arrays"""
    return np.maximum(0, x)

def get_all_relu_inputs(pth_file_path, batch_size, data_dir):
    """
    Captures input data for all nn.ReLU layers in the ResNet20 model.
    """
    model = resnet20()
    try:
        checkpoint = torch.load(pth_file_path, map_location=torch.device('cpu'))
        state_dict = checkpoint.get('net', checkpoint)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Successfully loaded model weights: {pth_file_path}")
    except Exception as e:
        print(f"Failed to load model weights: {e}")
        print("Please ensure your .pth file matches the ResNet20 model structure.")
        return None

    model.eval()

    all_relu_inputs = OrderedDict() # To store inputs for all ReLU layers
    relu_modules_info = OrderedDict() # To store a list of actual nn.ReLU modules

    # Register forward pre-hook to capture inputs of nn.ReLU layers
    def pre_hook_fn(module, input):
        # input is a tuple (input_tensor,) for a single input
        # We assume ReLU layers always have a single tensor input
        input_tensor = input[0].detach().cpu()
        all_relu_inputs[module.name].append(input_tensor)

    # Traverse all modules, find nn.ReLU, and register pre-hook
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.name = name # Assign a unique name for identification
            all_relu_inputs[name] = [] # Initialize list for this ReLU's inputs
            relu_modules_info[name] = module # Keep a reference to the module
            module.register_forward_pre_hook(pre_hook_fn)
    
    # Define data transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Starting forward pass on CIFAR-10 dataset to capture ReLU inputs. Total batches: {len(testloader)}")

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            _ = model(inputs)
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(testloader)} batches.")
            # Optional: Process only a subset of batches to reduce runtime/memory
            # if i >= 50: break

    print(f"Forward pass complete. Captured inputs for {len(all_relu_inputs)} ReLU layers.")

    # Log min/max values for each captured ReLU layer
    for relu_layer_name, inputs_list in all_relu_inputs.items():
        if inputs_list:
            all_inputs_tensor = torch.cat(inputs_list, dim=0)
            x_min = all_inputs_tensor.min().item()
            x_max = all_inputs_tensor.max().item()
            # Changed formatting to show full precision for x_min
            print(f"  ReLU Layer '{relu_layer_name}' captured input range: [{x_min}, {x_max:.4f}]") 
        else:
            print(f"  ReLU Layer '{relu_layer_name}' captured no inputs.")

    return all_relu_inputs

def plot_combined_relu_fit_for_layer(relu_layer_name, all_inputs_tensor, poly_coeffs, x_min, x_max, save_dir, relu_func):
    """
    Plots a combined figure of ReLU fits for all channels of a specific ReLU layer.
    """
    num_channels = all_inputs_tensor.shape[1]
    
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten() if num_channels > 1 else np.array([axes])

    poly_func = np.poly1d(poly_coeffs)
    
    # Define x_range for plotting purposes
    x_range_plot = np.linspace(x_min, x_max, 1000) # Use a dense range for smooth plotting
    y_true_plot = relu_func(x_range_plot)
    y_predicted_plot = poly_func(x_range_plot)
    mse = np.mean((y_true_plot - y_predicted_plot)**2) # Calculate MSE against true ReLU on full range

    for channel_idx in range(num_channels):
        ax = axes[channel_idx]
        channel_data = all_inputs_tensor[:, channel_idx, :, :].numpy().flatten()
        
        if len(channel_data) == 0:
            ax.set_title(f'Channel {channel_idx+1} (No Data)')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Use the global x_min, x_max for plotting, as the polynomial is fitted on that range
        ax.hist(channel_data, bins=50, density=True, alpha=0.3, color='lightgray')
        ax.plot(x_range_plot, y_true_plot, label='True ReLU', color='red', linestyle='--', linewidth=1)
        ax.plot(x_range_plot, y_predicted_plot, label=f'Fit (MSE:{mse:.2e})', color='blue', linewidth=1)
        ax.set_title(f'C{channel_idx+1} (MSE:{mse:.2e})', fontsize=9)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.grid(True, linestyle=':', alpha=0.6)
        if channel_idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    for i in range(num_channels, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle(f'Combined ReLU Fits for Layer: {relu_layer_name} (Poly Order: {len(poly_coeffs)-1})', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    filename = os.path.join(save_dir, f'{relu_layer_name.replace(".", "_")}_combined_relu_fits.jpg')
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    logging.info(f"  Saved combined ReLU fit plot for {relu_layer_name} to {filename}")


def polynomial_fit_relu_all_layers(args):
    """
    Performs polynomial fitting of the ReLU function for all nn.ReLU layers in the ResNet20 model,
    visualizes results, and saves polynomial coefficients.
    """
    os.makedirs(args.save_dir, exist_ok=True)

    log_file_path = os.path.join(args.save_dir, 'relu_fit_log.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                        format='%(asctime)s - %(message)s', filemode='w')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    if not logging.getLogger().handlers:
        logging.getLogger().addHandler(console_handler) 
    
    logging.info("--- ReLU Input Polynomial Fitting Log ---")
    logging.info(f"Model Path: {args.pth_file_path}")
    logging.info(f"Batch Size: {args.batch_size}")
    logging.info(f"Data Directory: {args.data_dir}")
    logging.info(f"Base Save Directory: {args.save_dir}")
    logging.info(f"Polynomial Order: {args.poly_order}\n")

    all_relu_inputs_raw = get_all_relu_inputs(
        args.pth_file_path, args.batch_size, args.data_dir
    )
    if all_relu_inputs_raw is None:
        logging.error("Could not capture ReLU inputs, terminating script.")
        return

    # Dictionary to store polynomial coefficients and associated info for all ReLU layers
    # Key: relu_layer_name (e.g., 'relu_initial', 'layer1.0.relu1', 'layer1.0.relu2')
    # Value: {'coeffs': [], 'min_val': float, 'max_val': float, 'mse': float, 'poly_order_used': int, 'status': str}
    all_poly_coeffs_info = OrderedDict() 

    for relu_layer_name, inputs_list in all_relu_inputs_raw.items():
        if not inputs_list:
            logging.warning(f"Warning: ReLU layer '{relu_layer_name}' captured no inputs. Skipping fitting.")
            continue
        
        # Concatenate all batch inputs for this ReLU layer
        # Shape: (total_samples, num_channels, height, width)
        all_inputs_tensor = torch.cat(inputs_list, dim=0) 
        num_channels = all_inputs_tensor.shape[1]
        
        # Flatten all data for this ReLU layer for fitting (treating all channels together for now)
        # This simplifies fitting a single polynomial for the entire layer's activation range.
        flat_input_data = all_inputs_tensor.numpy().flatten()

        if len(flat_input_data) == 0:
            logging.warning(f"Warning: ReLU layer '{relu_layer_name}' has no data. Skipping fitting.")
            continue

        x_min = np.min(flat_input_data)
        x_max = np.max(flat_input_data)

        if x_min == x_max:
            logging.warning(f"Warning: ReLU layer '{relu_layer_name}' input data is constant ({x_min:.4f}). Cannot fit polynomial. Skipping.")
            all_poly_coeffs_info[relu_layer_name] = {
                "coeffs": [],
                "min_val": float(x_min),
                "max_val": float(x_max),
                "mse": float('inf'),
                "poly_order_used": -1,
                "status": "constant_input"
            }
            continue 

        # Define x_range for MSE calculation and plotting
        # This should cover the observed range with sufficient points for accuracy
        x_range = np.linspace(x_min, x_max, 1000) # Use 1000 points for a smooth curve and accurate MSE

        # Generate evenly spaced points within the observed range for fitting the true ReLU function
        x_for_fit = np.linspace(x_min, x_max, 500) # 500 evenly distributed sample points for actual polyfit
        y_for_fit = relu(x_for_fit)

        current_poly_order = args.poly_order
        if current_poly_order >= len(x_for_fit):
            original_poly_order = current_poly_order
            current_poly_order = len(x_for_fit) - 1
            if current_poly_order < 0:
                logging.error(f"Error: ReLU layer '{relu_layer_name}' has insufficient points for fitting. Skipping.")
                all_poly_coeffs_info[relu_layer_name] = {
                    "coeffs": [],
                    "min_val": float(x_min),
                    "max_val": float(x_max),
                    "mse": float('inf'),
                    "poly_order_used": -1,
                    "status": "not_enough_points"
                }
                continue
            logging.warning(f"Warning: ReLU layer '{relu_layer_name}' polynomial order ({original_poly_order}) >= number of fit points ({len(x_for_fit)}). Adjusting order to {current_poly_order}.")
        
        try:
            poly_coeffs = np.polyfit(x_for_fit, y_for_fit, current_poly_order)
            poly_func = np.poly1d(poly_coeffs)
            y_predicted = poly_func(x_range) # Use defined x_range for MSE calculation
            mse = np.mean((relu(x_range) - y_predicted)**2) # Calculate MSE against true ReLU on full range

            # Store coefficients and related info
            all_poly_coeffs_info[relu_layer_name] = {
                "coeffs": poly_coeffs.tolist(), # Convert numpy array to list for JSON serialization
                "min_val": float(x_min),
                "max_val": float(x_max),
                "mse": float(mse),
                "poly_order_used": current_poly_order,
                "status": "success"
            }

        except Exception as e:
            logging.error(f"Error: ReLU layer '{relu_layer_name}' polynomial fit failed: {e}. Skipping.")
            all_poly_coeffs_info[relu_layer_name] = {
                "coeffs": [],
                "min_val": float(x_min),
                "max_val": float(x_max),
                "mse": float('inf'),
                "poly_order_used": -1,
                "status": f"fit_failed: {e}"
            }
            continue

        logging.info(f"--- Processing ReLU Layer: {relu_layer_name} (Channels: {num_channels}) ---")
        logging.info(f"  Input Range: [{x_min:.4f}, {x_max:.4f}], Fitted MSE: {mse:.6f}, Poly Order: {current_poly_order}")

        # Create a subdirectory for this ReLU layer's individual plots
        relu_layer_save_subdir = os.path.join(args.save_dir, relu_layer_name.replace(".", "_"))
        os.makedirs(relu_layer_save_subdir, exist_ok=True)
        logging.info(f"  Created subdirectory for {relu_layer_name}: {relu_layer_save_subdir}")

        # Plot and save the overall fit for this ReLU layer
        plt.figure(figsize=(10, 6))
        plt.hist(flat_input_data, bins=100, density=True, alpha=0.3, color='lightgray', 
                 label=f'ReLU Input Data Distribution (Min:{x_min:.2f}, Max:{x_max:.2f})')
        plt.plot(x_range, relu(x_range), label='True ReLU Function', color='red', linestyle='--', linewidth=2)
        plt.plot(x_range, y_predicted, label=f'Polynomial Fit (Order {current_poly_order})', color='blue', linewidth=2)
        plt.title(f'ReLU Polynomial Approximation for {relu_layer_name}\nMSE: {mse:.6f}')
        plt.xlabel('Input Value')
        plt.ylabel('Output Value / Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        individual_filename = os.path.join(relu_layer_save_subdir, f'{relu_layer_name.replace(".", "_")}_relu_fit.jpg')
        plt.savefig(individual_filename, dpi=200)
        plt.close()
        logging.info(f"  Saved individual ReLU fit plot to {individual_filename}")
        
        # Plot and save combined fit for all channels (if multi-channel)
        if num_channels > 1:
            # We are plotting the overall fit (one set of coeffs) across all channels
            # The coefficients are derived from the *flattened* data of all channels
            plot_combined_relu_fit_for_layer(relu_layer_name, all_inputs_tensor, poly_coeffs, x_min, x_max, args.save_dir, relu)

    # Save all polynomial coefficients to a JSON file
    poly_coeffs_json_path = os.path.join(args.save_dir, 'relu_poly_coeffs.json')
    with open(poly_coeffs_json_path, 'w') as f:
        json.dump(all_poly_coeffs_info, f, indent=4)
    logging.info(f"\nAll polynomial coefficients and info saved to: {poly_coeffs_json_path}")

    logging.info("\n--- ReLU Input Polynomial Fitting Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit ReLU function with a polynomial for all nn.ReLU layer inputs.')

    parser.add_argument('--pth_file_path', type=str, 
                        default="/home/wangtianyu/relu_finetune/base_models/20_ckpt_91.39_usp0.75.pth",
                        help='Path to the .pth model file.')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='Batch size for data loading.')
    parser.add_argument('--data_dir', type=str, default="./data",
                        help='Directory to store CIFAR-10 dataset.')
    parser.add_argument('--save_dir', type=str, 
                        default="/home/wangtianyu/relu_finetune/hook/all_relu_fits_input", 
                        help='Base directory to save the plots, log file, and polynomial coefficients.')
    parser.add_argument('--poly_order', type=int, default=63,
                        help='Order of the polynomial to fit the ReLU function.')

    args = parser.parse_args()

    polynomial_fit_relu_all_layers(args)

    print("\nAll nn.ReLU layer polynomial fits generated and saved to specified directory, along with log file and coefficients.")