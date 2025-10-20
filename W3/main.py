import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import initialize_phi, update_brightness, update_phi, calculate_difference

# ================================
# Input image
# ================================
folder_input = 'images/'
figure_name = 'circles.png'
img = cv2.imread(folder_input + figure_name, cv2.IMREAD_UNCHANGED)
img = img.astype('float')

# Visualize the grayscale image
cv2.imshow('Image', img); cv2.waitKey(0)

# Normalize image
img = (img.astype('float') - np.min(img))
img = img/np.max(img)
cv2.imshow('Normalized image', img)
cv2.waitKey(0)

# Make color images grayscale
# Skip this block if you handle the multi-channel Chan-Sandberg-Vese model
if len(img.shape) > 2:
    nc = img.shape[2] # number of channels
    img = np.mean(img, axis=2)

# ================================
# Parameters
# ================================
mu = 0.2
nu = 0
eta = 10e-8
lambda1 = 1
lambda2 = 1
tol = 10e-3
dt = 0.5
iterMax = int(1e5)

# ================================
# Phi initialization
# ================================

phi = initialize_phi(img.shape, method='checkerboard')

# Check initial phi
plt.figure(figsize=(5,4))
plt.imshow(phi, cmap='jet')
plt.colorbar(label=f'$\phi$ value')
plt.title('Initial level set $\phi$')
plt.show()

# ================================
# Main loop
# ================================

# CODE TO COMPLETE
# Explicit gradient descent or Semi-explicit (Gauss-Seidel) gradient descent (Bonus)
# Extra: Implement the Chan-Sandberg-Vese model (for colored images)
# Refer to Getreuer's paper (2012)

for it in range(iterMax):

    # Save previous phi
    phi_old = phi.copy()

    # Update brightness levels c1 and c2 with phi fixed
    c1, c2 = update_brightness(img, phi)

    # Update phi with c1 and c2 fixed
    phi = update_phi(phi, phi_old, c1, c2, img, mu, nu, eta, lambda1, lambda2, dt)

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