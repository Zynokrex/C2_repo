import cv2
import time
import csv
from datetime import datetime
from pathlib import Path

from utils import extract_seeds_from_mask, draw_seeds_on_image, parse_args
from image_graph import ImageGraph

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

    # Save the overlay (original image + seeds)
    overlay_display = draw_seeds_on_image(img, obj_seeds, bg_seeds)
    overlay_path = results_dir / f"{image_name}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_display)
    print(f"Saved seed overlay to {overlay_path}")

    # Run segmentation
    print("Building graph and running segmentation...")
    start_time = time.time()
    
    graph = ImageGraph(img, obj_seeds, bg_seeds, directed=args.directed, lambda_val=args.lambda_val, sigma_val=args.sigma_val) 
    flow, mask = graph.segment(algorithm=args.algorithm)
        
    end_time = time.time()
    seg_time = end_time - start_time
    print(f"Segmentation finished in {seg_time:.4f} seconds.")

    # Save segmentation result
    overlay_seg = img.copy()
    overlay_seg[mask == 0] = (0, 0, 255)  # mask == 0 is the object/source side. Red overlay on object
    result = cv2.addWeighted(img, 0.6, overlay_seg, 0.4, 0)
    segmented_path = results_dir / f"{image_name}_segmented.png"
    cv2.imwrite(str(segmented_path), result)
    print(f"Saved segmented image to {segmented_path}")

    # Save CSV with parameters and time
    csv_path = results_dir / "params_and_time.csv"
    report_data = {
        "image": args.image,
        "mask": args.mask,
        "directed": args.directed,
        "lambda": args.lambda_val,
        "sigma": args.sigma_val,
        "algorithm": args.algorithm,
        "segmentation_time_s": f"{seg_time:.4f}"
    }
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['parameter', 'value'])
        for key, value in report_data.items():
            writer.writerow([key, value])
    print(f"Saved parameters to {csv_path}")

if __name__ == "__main__":
    main()
