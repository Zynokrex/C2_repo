import cv2
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("./images/"),
                        help="Base folder containing image and mask files (default: ./images/)")
    parser.add_argument("--image", type=str, required=True,
                        help="Image filename (e.g., star.png). Searched under --data.")
    parser.add_argument("--mask", type=str, required=True,
                        help="Mask filename (e.g., star_mask.png). Searched under --data.")
    parser.add_argument("--lambda", type=float, default=1.0, dest="lambda_val",
                        help="Lambda value for regional cost.")
    parser.add_argument("--sigma", type=float, default=1.0, dest="sigma_val",
                        help="Sigma value for boundary cost.")
    args = parser.parse_args()

    # Build full paths from --data + filenames
    args.image = str(args.data / args.image)
    args.mask  = str(args.data / args.mask)
    return args


def extract_seeds_from_mask(mask_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts object (red) and background (blue) seed coordinates from a mask image.

    Args:
        mask_img (np.ndarray): The BGR mask image.
                               Red (0, 0, 255) = Object
                               Blue (255, 0, 0) = Background

    Returns:
        tuple[np.ndarray, np.ndarray]: (obj_seeds, bg_seeds)
                                       Arrays of (x, y) coordinates.
    """
    # Find pure red pixels [B=0, G=0, R=255]
    obj_mask = np.all(mask_img == [0, 0, 255], axis=-1)
    # Find pure blue pixels [B=255, G=0, R=0]
    bg_mask = np.all(mask_img == [255, 0, 0], axis=-1)

    # np.argwhere returns (row, col) which is (y, x)
    obj_coords_yx = np.argwhere(obj_mask)
    bg_coords_yx = np.argwhere(bg_mask)

    # Convert (y, x) to (x, y)
    obj_seeds_xy = obj_coords_yx[:, ::-1]
    bg_seeds_xy = bg_coords_yx[:, ::-1]

    return obj_seeds_xy, bg_seeds_xy

def draw_seeds_on_image(image: np.ndarray, obj_seeds: np.ndarray, bg_seeds: np.ndarray) -> np.ndarray:
    """
    Draws the object and background seeds directly as pure red / pure blue pixels.

    Args:
        image (np.ndarray): The original image.
        obj_seeds (np.ndarray): Array of (x, y) object seed coordinates.
        bg_seeds (np.ndarray): Array of (x, y) background seed coordinates.

    Returns:
        np.ndarray: A copy of the image with seeds drawn on it.
    """
    overlay = image.copy()

    obj_seeds_int = np.array(obj_seeds, dtype=np.int32)
    bg_seeds_int = np.array(bg_seeds, dtype=np.int32)

    # Paint object seeds (pure red, BGR = 0,0,255)
    for x, y in obj_seeds_int:
        if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
            overlay[y, x] = (0, 0, 255)

    # Paint background seeds (pure blue, BGR = 255,0,0)
    for x, y in bg_seeds_int:
        if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
            overlay[y, x] = (255, 0, 0)

    return overlay

def create_circle_image():
    """
    Creates a 256x256 white image with a circle of 128px diameter in the center.
    
    Returns:
        numpy.ndarray: The generated image as a BGR array
    """
    image = np.ones((256, 256, 3), dtype=np.uint8) * 255
    
    center = (128, 128)  # Center of the image
    radius = 64  # 128px diameter / 2 = 64px radius
    
    cv2.circle(image, center, radius, (0, 0, 0), -1)  # -1 for filled circle
    
    return image

def plot_image_with_seeds(image, obj_seeds, bg_seeds):
    overlay = image.copy()
    
    obj_seeds = np.array(obj_seeds, dtype=np.int32)
    bg_seeds = np.array(bg_seeds, dtype=np.int32)
    
    # Draw object seeds (red circles)
    for x, y in obj_seeds:
        cv2.circle(overlay, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red filled circle
        cv2.circle(overlay, (x, y), radius=6, color=(255, 255, 255), thickness=1)  # White border
    
    # Draw background seeds (blue squares)
    for x, y in bg_seeds:
        cv2.rectangle(overlay, (x-4, y-4), (x+4, y+4), color=(255, 0, 0), thickness=-1)  # Blue filled square
        cv2.rectangle(overlay, (x-5, y-5), (x+5, y+5), color=(255, 255, 255), thickness=1)  # White border
    
    # Display with CV2
    cv2.imshow("Overlay", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()