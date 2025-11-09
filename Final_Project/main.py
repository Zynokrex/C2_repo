import cv2
import time
import csv
from datetime import datetime
from pathlib import Path
import numpy as np

from utils import extract_seeds_from_mask, draw_seeds_on_image, parse_args
from image_graph import ImageGraph

BASE_RESULTS_DIR = Path("./results/")

class Segmenter:
    """
    Encapsulates running ImageGraph segmentation and writing standardized outputs.

    Returns:
      dict with keys: flow, mask (HxW uint8), result_rgb (HxW uint8), seg_time, paths (dict)
    """
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)

    def _make_results_dir(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _save_csv(self, out_dir: Path, params: dict):
        csv_path = out_dir / "params_and_time.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["parameter", "value"])
            for k, v in params.items():
                w.writerow([k, v])
        return csv_path

    def _save_image_obj(self, out_path: Path, img_obj: np.ndarray):
        cv2.imwrite(str(out_path), img_obj)
    
    def _save_overlay(self, out_path: Path, img: np.ndarray, obj_seeds: np.ndarray, bg_seeds: np.ndarray):
        overlay_img = draw_seeds_on_image(img, obj_seeds, bg_seeds)
        cv2.imwrite(str(out_path), overlay_img)

    def run(self, img: np.ndarray, obj_seeds: np.ndarray, bg_seeds: np.ndarray,
            lambda_val: float = 1.0, sigma_val: float = 1.0, algorithm: str = "bk",
            image_name: str = "image", save: bool = True) -> dict:
        # Validate input shapes & types
        if img is None:
            raise ValueError("image_bgr is None")
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # choose results folder
        out_dir = self.results_dir
        
        # Start timing
        start = time.time()

        # Run segmentation
        graph = ImageGraph(img, obj_seeds, bg_seeds, lambda_val=lambda_val, sigma_val=sigma_val)
        flow, mask = graph.segment(algorithm=algorithm)

        seg_time = time.time() - start

        # Create visualization (BGR)
        overlay_seg = img.copy()
        overlay_seg[mask == 0] = (0, 0, 255)  # object red (BGR)
        result_obj = cv2.addWeighted(img, 0.6, overlay_seg, 0.4, 0)

        # Save outputs
        paths = {}
        if save:
            overlay_path = out_dir / f"{image_name}_overlay.png"
            segmented_path = out_dir / f"{image_name}_segmented.png"
            # Save original overlay and segmented
            self._save_overlay(overlay_path, img, obj_seeds, bg_seeds)
            self._save_image_obj(segmented_path, result_obj)
            # Save params CSV
            params = {
                "image": image_name,
                "lambda": lambda_val,
                "sigma": sigma_val,
                "algorithm": algorithm,
                "segmentation_time_s": f"{seg_time:.4f}"
            }
            csv_path = self._save_csv(out_dir, params)
            paths.update({"overlay": str(overlay_path), "segmented": str(segmented_path), "csv": str(csv_path)})

        # Convert result for display (RGB)
        result_rgb = result_obj[..., ::-1]

        return {
            "flow": float(flow),
            "mask": mask,
            "result_rgb": result_rgb,
            "seg_time": seg_time,
            "paths": paths
        }

def main():
    args = parse_args()

    print(f"Image: {args.image}")
    print(f"Mask: {args.mask}")
    print(f"Directed: {args.directed}")
    print(f"Lambda: {args.lambda_val}")
    print(f"Sigma: {args.sigma_val}")
    print(f"Algorithm: {args.algorithm}")

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(args.image).stem
    results_dir = Path("./results") / f"exp_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load image and mask
    img_path = Path(args.image)
    mask_path = Path(args.mask)
    img = cv2.imread(str(img_path))
    mask_img = cv2.imread(str(mask_path))

    if img is None:
        print(f"Error: Could not read image from {img_path}")
        return
    if mask_img is None:
        print(f"Error: Could not read mask from {mask_path}")
        return
    if img.shape[:2] != mask_img.shape[:2]:
        print(f"Error: Image shape {img.shape[:2]} and mask shape {mask_img.shape[:2]} do not match.")
        return

    # Extract seeds
    obj_seeds, bg_seeds = extract_seeds_from_mask(mask_img)
    print(f"Found {len(obj_seeds)} object seeds and {len(bg_seeds)} background seeds.")
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = Path(args.image).stem


    segmenter = Segmenter(results_dir=results_dir)  # or pass results_dir explicitly
    segmenter.run(img, obj_seeds, bg_seeds, lambda_val=args.lambda_val,
                        sigma_val=args.sigma_val, algorithm=args.algorithm,
                        image_name=image_name, save=True)



if __name__ == "__main__":
    main()
