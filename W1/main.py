import cv2
import numpy as np
from dataclasses import dataclass
import inpainting
import os
import argparse

@dataclass
class Parameters:
    hi: float
    hj: float

# Choose the image name via CLI
parser = argparse.ArgumentParser(description="Inpaint an image by name (e.g., image7).")
parser.add_argument("--image", nargs="?", required=True)
args = parser.parse_args()
image_name = args.image

# Folder with the images
image_folder = 'images/'
results_folder = 'results/'
os.makedirs(results_folder, exist_ok=True)

# Helper to find an image trying different extensions
def find_with_ext(base_path_wo_ext, exts):
    for ext in exts:
        path = base_path_wo_ext + ext
        if os.path.exists(path):
            return path
    return None

img_path = find_with_ext(
    os.path.join(image_folder, image_name + '_to_restore'),
    ['.png', '.tif', '.jpg']
)
mask_path = find_with_ext(
    os.path.join(image_folder, image_name + '_mask'),
    ['.png', '.tif', '.jpg']
)

# Read the image to be restored
im = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# Print image dimensions
print('Image Dimensions : ', im.shape)
print('Image Height     : ', im.shape[0])
print('Image Width      : ', im.shape[1])

# Normalize values into [0,1]
min_val = np.min(im)
max_val = np.max(im)
im = (im.astype('float') - min_val)
im = im / max_val

# Load the mask image
mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

# From the mask image we define a binary mask that
# "erases" the darker pixels from the original image
mask = (mask_img > 128).astype('float')

# Mask dimensions
dims = mask.shape
ni = mask.shape[0]
nj = mask.shape[1]
print('Mask Dimension : ', dims)
print('Mask Height    : ', ni)
print('Mask Width     : ', nj)

# Parameters
param = Parameters(0, 0)
param.hi = 1 / (ni - 1)
param.hj = 1 / (nj - 1)

# Check whether the image is gray or color and apply the corresponding pipeline
if im.ndim == 2:
    # Grayscale
    print('Processing a grayscale image')
    u = inpainting.laplace_equation(im, mask, param)
else:
    # Color (process each channel separately)
    print('Processing a color image')
    if mask.ndim == 2:
        # If mask is single-channel but image is multi-channel, replicate mask
        mask = np.repeat(mask[:, :, None], im.shape[2], axis=2)

    u = np.zeros_like(im, dtype=float)
    for c in range(im.shape[2]):
        u[:, :, c] = inpainting.laplace_equation(im[:, :, c], mask[:, :, c], param)

# Save the inpainted image
out_path = os.path.join(results_folder, f'{image_name}_inpainted.png')
# Clip to [0,1] then scale to 8-bit for saving
u_to_save = np.clip(u, 0, 1)
u_to_save = (u_to_save * 255).astype(np.uint8)
cv2.imwrite(out_path, u_to_save)
print(f'Saved: {out_path}')
