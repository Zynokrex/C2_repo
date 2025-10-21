import numpy as np

def initialize_phi(shape, method='checkerboard', seed=42):

    ni, nj = shape
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

def update_brightness(img, phi, epsilon=1):
    """
    Compute the average brightness values inside and outside a contour.

    Args:
        img: input image
        phi: level set function
        epsilon: smoothing parameter
    """
    H_phi = heavyside(phi, epsilon)

    c1 = np.sum(H_phi * img) / np.sum(H_phi)
    c2 = np.sum((1 - H_phi) * img) / np.sum(1 - H_phi)

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

def update_phi(phi, phi_old, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt, epsilon=1):

    # Update countours (mirror)
    phi = update_exterior(phi)
    phi_old = update_exterior(phi_old)

    # Update interior with Gauss-Seidel
    phi = update_interior(phi_old, phi, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt, epsilon)

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

