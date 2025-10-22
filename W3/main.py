import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import initialize_phi, update_brightness, update_phi, calculate_difference

def main(args):
    folder_input = args.input_folder
    figure_name = args.image

    # ================================
    # Input image
    # ================================
    img = cv2.imread(os.path.join(folder_input, figure_name), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {os.path.join(folder_input, figure_name)}")
    img = img.astype('float')

    # Visualize the grayscale image
    cv2.imshow('Image', img); cv2.waitKey(0)

    # Normalize image
    img = (img.astype('float') - np.min(img))
    img = img/np.max(img)
    cv2.imshow('Normalized image', img)
    cv2.waitKey(0)

    # ================================
    # Detect color mode instead of forcing grayscale
    # ================================
    if len(img.shape) > 2:
        color_mode = "Colored"
        print(f"Running Chan-Vese in 'Colored' mode.")
    else:
        color_mode = "Gray"
        print("Running Chan-Vese in 'Gray' mode.")

    # ================================
    # Parameters (from args)
    # ================================
    mu = float(args.mu)
    nu = float(args.nu)
    eta = float(args.eta)
    lambda1 = float(args.lambda1)
    lambda2 = float(args.lambda2)
    tol = float(args.tol)
    dt = float(args.dt)
    iterMax = int(args.iterMax)
    epsilon = float(args.epsilon)

    # ================================
    # Phi initialization
    # ================================
    phi = initialize_phi(img.shape, method='checkerboard', color=color_mode)

    # Check initial phi
    plt.figure(figsize=(5,4))
    plt.imshow(phi, cmap='jet')
    plt.colorbar(label=r'$\phi$ value')
    plt.title(r'Initial level set $\phi$')
    plt.show()

    # ================================
    # Main loop
    # ================================
    for it in range(iterMax):

        # Save previous phi
        phi_old = phi.copy()

        # Update brightness levels c1 and c2 with phi fixed
        c1, c2 = update_brightness(img, phi, color=color_mode)

        # Update phi with c1 and c2 fixed
        phi = update_phi(phi, phi_old, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt, epsilon=epsilon, color=color_mode)

        # Check for convergence
        diff = calculate_difference(phi, phi_old)
        print(diff)
        if diff <= tol:
            print(f'Converged in {it} iterations.')
            break

    # Segmented image 
    seg = (phi >= 0).astype(np.uint8)*255

    # Show output image
    cv2.imshow('Segmented image', seg); 
    cv2.waitKey(0)

    # Save outputs
    results_folder = args.results_folder
    os.makedirs(results_folder, exist_ok=True)
    
    # Build parameter string to append to filenames
    param_str = f"mu{mu}_nu{nu}_eta{eta}_l1{lambda1}_l2{lambda2}_dt{dt}_eps{epsilon}_it{iterMax}"

    base_name, ext = os.path.splitext(figure_name)
    seg_path = os.path.join(results_folder, f'seg_{base_name}_{param_str}{ext}')
    cv2.imwrite(seg_path, seg)
    print(f"Saved segmented image to: {seg_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='circles.png', help='Image filename to be processed')
    parser.add_argument('--input_folder', type=str, default='images', help='Folder containing input images')
    parser.add_argument('--results_folder', type=str, default='results/', help='Folder to save results')
    # Poisson / Chan-Vese parameters (allow overrides from CLI)
    parser.add_argument('--mu', type=float, default=0.2, help='Mu (length term weight)')
    parser.add_argument('--nu', type=float, default=0.0, help='Nu (area term weight)')
    parser.add_argument('--eta', type=float, default=1e-8, help='Eta (small constant)')
    parser.add_argument('--lambda1', type=float, default=1.0, help='Lambda1 (fidelity weight inside)')
    parser.add_argument('--lambda2', type=float, default=1.0, help='Lambda2 (fidelity weight outside)')
    parser.add_argument('--tol', type=float, default=1e-2, help='Convergence tolerance')
    parser.add_argument('--dt', type=float, default=0.5, help='Time step dt')
    parser.add_argument('--iterMax', type=int, default=100000, help='Maximum number of iterations')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Epsilon for regularization')
    args = parser.parse_args()

    main(args)
