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

def update_brightness(img, phi, epsilon):
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


def update_interior(phi, c1, c2, img, mu, nu, lambda1, lambda2, dt):

    return phi


def update_phi(phi, c1, c2, img, mu, nu, lambda1, lambda2, dt):

    # Update countours (mirror)
    phi = update_exterior(phi)

    # Update interior with Gauss-Seidel
    phi = update_interior(phi, c1, c2, img, mu, nu, lambda1, lambda2, dt)

    return phi


def calculate_difference(phi, phi_old):

    diff = 0

    return diff