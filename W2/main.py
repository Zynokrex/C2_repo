import cv2
import numpy as np
import poisson_editing
from scipy.sparse.linalg import LinearOperator, cg

def shift_source(src, src_masks, translations, ni, nj, nChannels, padding=5):
    translated_image = src.copy()
    for t, src_mask in zip(translations, src_masks):
        dy, dx = t
        for c in range(nChannels):
            mask_channel = src_mask[:, :, c] // 255

            # Find indices (points) where the mask is active
            indices = np.where(mask_channel != 0)
            if indices[0].size == 0:
                continue

            i_min, i_max = indices[0].min(), indices[0].max()
            j_min, j_max = indices[1].min(), indices[1].max()

            top    = max(0, i_min - padding)
            bottom = min(ni, i_max + 1 + padding)   # +1: inclusive -> exclusive
            left   = max(0, j_min - padding)
            right  = min(nj, j_max + 1 + padding)

            # Build full rectangular grid of the crop region
            rect_i = np.arange(top, bottom, dtype=np.int32)
            rect_j = np.arange(left, right, dtype=np.int32)
            grid_i, grid_j = np.meshgrid(rect_i, rect_j, indexing='ij')

            # Shifted positions
            shifted_i = grid_i + dy
            shifted_j = grid_j + dx

            # Only keep indices inside image bounds
            valid = (shifted_i >= 0) & (shifted_i < ni) & (shifted_j >= 0) & (shifted_j < nj)

            # Paste the rectangular crop (including padding/context)
            translated_image[shifted_i[valid], shifted_j[valid], c] = src[grid_i[valid], grid_j[valid], c]
    return translated_image


# Load images
src = cv2.imread('images/lena/girl.png').astype(np.float64)
dst = cv2.imread('images/lena/lena.png').astype(np.float64)
# For Mona Lisa and Ginevra:
# src = cv2.imread('images/monalisa/ginevra.png')
# dst = cv2.imread('images/monalisa/monalisa.png')

# Customize the code with your own pictures and masks.

# Store shapes and number of channels (src, dst and mask should have same dimensions!)
ni, nj, nChannels = dst.shape

# Display the images
cv2.imshow('Source image', src); cv2.waitKey(0)
cv2.imshow('Destination image', dst); cv2.waitKey(0)

# Load masks for eye swapping
src_mask_eyes = cv2.imread('images/lena/mask_src_eyes.png', cv2.IMREAD_COLOR)
dst_mask_eyes = cv2.imread('images/lena/mask_dst_eyes.png', cv2.IMREAD_COLOR)
cv2.imshow('Eyes source mask', src_mask_eyes); cv2.waitKey(0)
cv2.imshow('Eyes destination mask', dst_mask_eyes); cv2.waitKey(0)

# Load masks for mouth swapping
src_mask_mouth = cv2.imread('images/lena/mask_src_mouth.png', cv2.IMREAD_COLOR)
dst_mask_mouth = cv2.imread('images/lena/mask_dst_mouth.png', cv2.IMREAD_COLOR)
cv2.imshow('Mouth source mask', src_mask_mouth); cv2.waitKey(0)
cv2.imshow('Mouth destination mask', dst_mask_mouth); cv2.waitKey(0)

# Get the translation vectors (hard coded)
t_eyes = poisson_editing.get_translation(src_mask_eyes, dst_mask_eyes, "eyes")
t_mouth = poisson_editing.get_translation(src_mask_mouth, dst_mask_mouth, "mouth")

translations = [t_eyes, t_mouth]
src_masks = [src_mask_eyes, src_mask_mouth]
translated_image = shift_source(src, src_masks, translations, ni, nj, nChannels)
cv2.imshow('Source image after shifting', translated_image); cv2.waitKey(0)

u_comb = dst.copy()
mask = np.zeros_like(dst) # combined mask
# Blend with the original (destination) image
for dst_mask in [dst_mask_eyes, dst_mask_mouth]:
    for c in range(nChannels):
        mask_channel = dst_mask[:, :, c] // 255  # Convert to binary mask
        mask[:, :, c] = np.where(mask_channel != 0, 1, mask[:, :, c])
        u_comb[:, :, c] = np.where(mask_channel != 0, translated_image[:, :, c], u_comb[:, :, c])
cv2.imshow('Blended image before Poisson editing', u_comb); cv2.waitKey(0)

u_comb = np.zeros_like(dst) # combined image

for channel in range(3):

    m = mask[:, :, channel]
    f = dst[:, :, channel]
    u1 = translated_image[:, :, channel]

    beta_0 = 1   # TRY CHANGING
    beta = beta_0 * (1 - m)

    vi, vj = poisson_editing.composite_gradients(u1, f, m)
    b = beta * f - poisson_editing.divergence(vi, vj)
    
    ni, nj = f.shape
    def matvec(x):
        u_img = x.reshape(ni, nj)
        Au = poisson_editing.poisson_linear_operator(u_img, beta)
        return Au.ravel()
    
    A = LinearOperator((ni*nj, ni*nj), matvec=matvec, dtype=np.float64)

    x0 = f.ravel()
    x, info = cg(A, b.ravel(), x0=x0, atol=1e-6, maxiter=800)

    u_comb[:, :, channel] = x.reshape(ni, nj)

u_final = np.clip(u_comb, 0, 255).astype(np.uint8)

cv2.imshow('Final result of Poisson blending', u_final); cv2.waitKey(0)
