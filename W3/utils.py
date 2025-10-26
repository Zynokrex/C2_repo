import numpy as np
import matplotlib.pyplot as plt

def initialize_phi(shape, method='checkerboard', seed=42, color="Gray"):

    if color == "Gray":
        ni, nj = shape
    else:
        ni, nj = shape[0], shape[1]

    Y, X = np.meshgrid(np.arange(ni), np.arange(nj), indexing='ij')

    # Initialize phi based on the selected method
    if method == 'circle': # Teacher's initial suggestion
        phi = (-np.sqrt((X - np.round(ni / 2)) ** 2 + (Y - np.round(nj / 2)) ** 2) + 50)
    elif method == 'checkerboard': # Pascal Getreuer's checkerboard
        phi = np.sin(np.pi * X / 5) * np.sin(np.pi * Y / 5) 
    elif method == 'random':
        np.random.seed(seed)
        phi = 2 * np.random.rand(ni, nj) - 1
    else:
        raise ValueError(f"Unknown initialization method '{method}'.")

    # Normalize to [-1, 1]
    phi = (phi - phi.min()) / (phi.max() - phi.min())
    phi = 2*phi - 1

    return phi

def heavyside(phi, epsilon=1):
    """
    Smoother heavyside function
    Args:
        phi: input
        epsilon: smoothing parameter

    """
    return 0.5 * (1 + (2 / np.pi) * np.arctan(phi / epsilon))

def dirac(t, epsilon=1):
    return epsilon / (np.pi * (np.pow(epsilon, 2) + np.pow(t, 2)))

def update_brightness(img, phi, epsilon=1, color="Gray"):
    """
    Compute the average brightness values inside and outside a contour.

    Args:
        img: input image
        phi: level set function
        epsilon: smoothing parameter
        color: "Gray" or "Colored"
    """
    H_phi = heavyside(phi, epsilon)

    if color == "Colored":
        # For colored images, compute c1 and c2 for each channel
        c1 = np.zeros(img.shape[2])
        c2 = np.zeros(img.shape[2])
        for ch in range(img.shape[2]):
            c1[ch] = np.sum(H_phi * img[:, :, ch]) / (np.sum(H_phi) + 1e-8)
            c2[ch] = np.sum((1 - H_phi) * img[:, :, ch]) / (np.sum(1 - H_phi) + 1e-8)
    else:
        # For grayscale images, compute scalar c1 and c2
        c1 = np.sum(H_phi * img) / (np.sum(H_phi) + 1e-8)
        c2 = np.sum((1 - H_phi) * img) / (np.sum(1 - H_phi) + 1e-8)

    return c1, c2


def update_exterior(phi):
    """
    Enforce Neumann boundary conditions by duplicating
    pixels near the borders.

    Args:
        phi: level set function
    """
    phi[0, :]   = phi[1, :]     # Top boundary
    phi[-1, :]  = phi[-2, :]    # Bottom boundary
    phi[:, 0]   = phi[:, 1]     # Left boundary
    phi[:, -1]  = phi[:, -2]    # Right boundary

    return phi


def update_interior(phi_n, phi_np1, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt, epsilon=1, color="Gray"):
    """
    color: Can be Gray for grayscale img or Colored for imgs with more than 2 channels
    """
    #We add an argument to see if we will use Colored or Grayscale images

    rows, cols = phi_n.shape

    # Gauss-Seidel sweep: iterate from left-to-right, top-to-bottom
    # We loop over the inner region
    for i in range(1, rows - 1): # Exclude first and last row
        for j in range(1, cols - 1): # Exclude first and last column
            
            # --- Calculate A and B coefficients for the current point ---
            A_ij = calculate_A_at_point(phi_n, phi_np1, mu, eta, i, j)
            B_ij = calculate_B_at_point(phi_n, phi_np1, mu, eta, i, j)
            
            # --- Get A and B values from neighbors ---
            A_prev_i_j = calculate_A_at_point(phi_n, phi_np1, mu, eta, i-1, j)
            B_i_prev_j = calculate_B_at_point(phi_n, phi_np1, mu, eta, i, j-1)
            
            # Get f_i,j from image data
            f_ij = img[i, j]

            # Calculate delta_epsilon(phi_i,j^n)
            delta_val = dirac(phi_n[i, j], epsilon)

            # --- Numerator terms ---
            term1 = phi_n[i, j]
            
            term2_coeff = dt * delta_val
            
            term3 = (A_ij * phi_n[i+1, j] + 
                                  A_prev_i_j * phi_np1[i-1, j] + 
                                  B_ij * phi_n[i, j+1] + 
                                  B_i_prev_j * phi_np1[i, j-1])
            
            # Region fitting terms. 
            if color == "Gray": 
                region_fitting_term = - nu  - lambda1 * (f_ij - c1)**2 + lambda2 * (f_ij - c2)**2
            
            elif color == "Colored":
                region_fitting_term = - nu  - lambda1 * np.linalg.norm(f_ij - c1)**2 + lambda2 * np.linalg.norm(f_ij - c2)**2

            numerator_bracket = term3 + region_fitting_term
            
            numerator = term1 + term2_coeff * numerator_bracket

            # --- Denominator terms ---
            denominator_sum = (1 + dt * delta_val * (A_ij + A_prev_i_j + B_ij + B_i_prev_j))
            
            # Update phi_np1 at (i,j)
            phi_np1[i, j] = numerator / denominator_sum
            
    return phi_np1

def update_phi(phi, phi_old, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt, epsilon=1, color="Gray"):

    # Update countours (mirror)
    phi = update_exterior(phi)
    phi_old = update_exterior(phi_old)

    # Update interior with Gauss-Seidel
    phi = update_interior(phi_old, phi, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt, epsilon, color)

    return phi


def calculate_difference(phi, phi_old):

    diff = phi - phi_old
    l2_norm_diff = np.linalg.norm(diff)

    return l2_norm_diff

def calculate_A_at_point(phi_n, phi_np1, mu, eta, i, j): 
    denominator = np.pow(eta, 2) + np.pow((phi_n[i+1,j] - phi_n[i,j]), 2) + np.pow(((phi_n[i,j+1] - phi_np1[i,j-1])/2), 2) 
    return mu / np.sqrt(denominator)

def calculate_B_at_point(phi_n, phi_np1, mu, eta, i, j): 
    denominator = np.pow(eta, 2) + np.pow(((phi_n[i+1,j] - phi_np1[i-1,j])/2), 2) + np.pow((phi_n[i,j+1] - phi_n[i,j]), 2) 
    return mu / np.sqrt(denominator)

def save_overlay(img, phi, out_path, color_mode):
    plt.figure(figsize=(6, 6))
    if color_mode == "Colored":
        bg = img.copy()
        plt.imshow(bg[..., ::-1])   # BGR->RGB
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.contour(phi, levels=[0], linewidths=1.5, colors='r')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()


def plot_diff_progress(values, title="Diff Progress", ylabel="Diff value", xlabel="Step", save_path=None):

    plt.figure(figsize=(8, 5))
    plt.plot(values, marker='o', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to: {save_path}")

    plt.show()