import cv2
import numpy as np

img_folder = 'images/'
img_path = img_folder + 'image7_to_restore.jpg'
img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# Convert to float [0,1] for Lab conversion
img_rgb_f = img_rgb.astype(np.float32) / 255.0
img_lab = cv2.cvtColor(img_rgb_f, cv2.COLOR_RGB2LAB)

# We are looking for a bright color where the red channel is significantly
# higher than the green and blue channels. Pre-filtering.
r = img_rgb[:,:,0]
g = img_rgb[:,:,1]
b = img_rgb[:,:,2]
mask_candidate = (r > 150) & (r > g*1.5) & (r > b*1.5)
mask_candidate_img = (mask_candidate.astype(np.uint8)) * 255

cv2.imwrite(img_folder + "image7_mask_candidate.jpg", mask_candidate_img)
candidate_lab = img_lab[mask_candidate]

if candidate_lab.shape[0] == 0:
    raise ValueError("No red candidates found for reference selection.")

ref_lab = np.median(candidate_lab, axis=0)
dist_map = np.sqrt(np.sum((img_lab - ref_lab)**2, axis=2))

# Keep pixels whose distance is <= 95th percentile of red candidate distances
ref_dists = np.sqrt(np.sum((candidate_lab - ref_lab)**2, axis=1))
percentile = 95
threshold = np.percentile(ref_dists, percentile)
mask_dist = (dist_map <= threshold).astype(np.uint8) * 255
cv2.imwrite(img_folder + "image7_mask_distance.jpg", mask_dist)

# Reduce noise with a closing, in order to fill small holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask_clean = cv2.morphologyEx(mask_dist, cv2.MORPH_CLOSE, kernel, iterations=1)

mask_out_path = img_folder + "image7_mask.jpg"
cv2.imwrite(str(mask_out_path), mask_clean)
