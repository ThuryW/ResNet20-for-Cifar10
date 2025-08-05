import numpy as np
import json
from numpy.polynomial import chebyshev
import matplotlib.pyplot as plt
import os

def fit_relu_with_chebyshev(degree, domain_min=-10, domain_max=10, num_points=1000, save_path='chebyshev_results.json'):
    """
    Fits the ReLU function with a Chebyshev polynomial, evaluates the accuracy,
    and saves the coefficients and the plot to a specified path.
    """
    # 1. Define the ReLU function
    def relu(x):
        return np.maximum(0, x)

    # 2. Generate sample points for fitting
    x_normalized = chebyshev.chebpts1(degree + 1)
    x_cheb = 0.5 * (domain_max - domain_min) * x_normalized + 0.5 * (domain_max + domain_min)
    y_cheb = relu(x_cheb)

    # 3. Compute Chebyshev polynomial coefficients
    cheb_coeffs = chebyshev.chebfit(x_normalized, y_cheb, degree)
    print(f"Chebyshev polynomial coefficients (degree {degree}):")
    print(cheb_coeffs)

    # 4. Save polynomial coefficients to a JSON file
    coefficients_list = cheb_coeffs.tolist()
    with open(save_path, 'w') as f:
        json.dump(coefficients_list, f, indent=4)
    print(f"\nPolynomial coefficients successfully saved to: {save_path}")

    # 5. Evaluate fitting accuracy
    x_eval = np.linspace(domain_min, domain_max, num_points)
    y_true = relu(x_eval)
    x_eval_normalized = (2 * x_eval - (domain_min + domain_max)) / (domain_max - domain_min)
    y_fit = chebyshev.chebval(x_eval_normalized, cheb_coeffs)

    absolute_error = np.abs(y_true - y_fit)
    relative_error = np.zeros_like(y_true)
    non_zero_mask = y_true != 0
    relative_error[non_zero_mask] = np.abs((y_true[non_zero_mask] - y_fit[non_zero_mask]) / y_true[non_zero_mask])

    mse = np.mean(absolute_error ** 2)
    max_absolute_error = np.max(absolute_error)
    max_relative_error = np.max(relative_error)

    print("\n--- Fitting Accuracy Evaluation ---")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Maximum Absolute Error: {max_absolute_error:.6f}")
    print(f"Maximum Relative Error: {max_relative_error:.6f}")

    # 6. Visualize and save the image
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x_eval, y_true, label='ReLU Function', color='blue', linewidth=2)
    plt.plot(x_eval, y_fit, label=f'Chebyshev Polynomial Fit (Degree={degree})', color='red', linestyle='--')
    plt.title(f'Chebyshev Polynomial Fit of ReLU Function (Degree={degree})')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    
    # Construct the image save path
    dir_name = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    image_path = os.path.join(dir_name, f'{base_name}.jpg')
    
    # Save the plot
    fig.savefig(image_path, bbox_inches='tight', dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"\nFitting plot successfully saved to: {image_path}")

# --- Example Usage ---
if __name__ == '__main__':
    poly_degree = 31
    # Set the save path, you can specify a directory
    output_path = '/home/wangtianyu/relu_finetune/scripts/single_relu/relu_coeffs.json'
    
    # Ensure the save directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creating directory: {output_dir}")

    fit_relu_with_chebyshev(degree=poly_degree, save_path=output_path)