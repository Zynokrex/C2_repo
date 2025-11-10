from pathlib import Path
import time

import gradio as gr
import cv2
import numpy as np

from main import Segmenter
from utils import extract_seeds_from_mask

# Callback that runs segmentation using stored seeds + selected params
def run_segmentation(original_rgb, obj_seeds, bg_seeds, lambda_val, sigma_val, algorithm, directed):
    # original_rgb: RGB uint8 image captured from the editor's background via gr.State
    if original_rgb is None:
        return "No image provided", None

    # Convert to BGR for ImageGraph (it expects BGR and will convert to grayscale internally)
    image_bgr = original_rgb[..., ::-1].astype(np.uint8)

    # Validate seeds (KDE requires >= 2 per class)
    if obj_seeds is None or bg_seeds is None or len(obj_seeds) < 2 or len(bg_seeds) < 2:
        return "Need at least 2 seeds for each class", None

    try:
        # Prepare results directory per run and call Segmenter to ensure
        # consistent timing, visualization and CSV saving as in CLI.
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("results") / f"gradio_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        segmenter = Segmenter(results_dir=out_dir)
        # Use a generic image name (no extension) for saved artifacts
        res = segmenter.run(
            img=image_bgr,
            obj_seeds=obj_seeds,
            bg_seeds=bg_seeds,
            lambda_val=lambda_val,
            sigma_val=sigma_val,
            algorithm=algorithm,
            directed=directed,
            image_name="gradio"
        )

        msg = (
            f"Segmentation done (flow={res['flow']:.2f}, time={res['seg_time']:.3f}s).\n"
            f"Saved: {res['paths']}"
        )
        return msg, res["result_rgb"]
    except Exception as e:
        return f"Segmentation failed: {e}", None

# Helper: normalize editor composite -> exact BGR uint8 mask and compute seeds
def map_to_allowed_seed_colors(rgb: np.ndarray, l2_threshold: int = 30) -> np.ndarray:
    """Map arbitrary RGB pixels to three classes: nothing, blue seed, red seed.

    Only pixels within an L2 distance threshold of the allowed colors are mapped
    to those exact colors; all other pixels are left as 'nothing' (set to black).
    """
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)

    allowed_rgb = np.array([
        [0, 0, 255],   # Blue background
        [255, 0, 0],   # Red object
    ], dtype=np.uint8)

    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3).astype(np.int32)
    allowed = allowed_rgb.astype(np.int32)
    dists_sq = np.sum((pixels[:, None, :] - allowed[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dists_sq, axis=1)
    min_dist_sq = dists_sq[np.arange(dists_sq.shape[0]), nearest]

    # Only accept as seed if within the L2 threshold
    thr_sq = int(l2_threshold) * int(l2_threshold)
    accepted = min_dist_sq <= thr_sq

    # Start with 'nothing' (black) everywhere, then fill accepted seeds
    mapped_rgb = np.zeros_like(pixels, dtype=np.uint8)
    mapped_rgb[accepted] = allowed[nearest[accepted]].astype(np.uint8)
    mapped_rgb = mapped_rgb.reshape(h, w, 3)
    return mapped_rgb


def process_mask_and_store(img_editor_output):
    composite = img_editor_output["composite"]  # RGB(A) uint8
    bg_rgb = img_editor_output.get("background", None)
    if composite.shape[2] == 4:
        rgb = composite[..., :3]
    else:
        rgb = composite

    mapped_rgb = map_to_allowed_seed_colors(rgb)

    # Convert to BGR for seed extraction only (not stored in state)
    bgr_mask = mapped_rgb[..., ::-1]
    obj_seeds, bg_seeds = extract_seeds_from_mask(bgr_mask)

    # Ensure background image is valid RGB (fallback to composite if missing)
    if bg_rgb is None:
        bg_rgb = rgb
    elif bg_rgb.ndim == 3 and bg_rgb.shape[2] == 4:
        bg_rgb = bg_rgb[..., :3]

    info = f"Stored seeds: obj={len(obj_seeds)} bg={len(bg_seeds)}; image shape={bg_rgb.shape}"

    return info, obj_seeds.astype(np.int32), bg_seeds.astype(np.int32), bg_rgb


# Only colors available in the editor
seed_colors = [
    "#FF0000",  # Red  (for Object)     - RGB hex
    "#0000FF",  # Blue (for Background) - RGB hex
]

with gr.Blocks() as demo:
    gr.Markdown("# C2 — Graph-Cut Segmentation Demo")
    # Use gr.ImageEditor
    editor = gr.ImageEditor(
        type="numpy",
        label="Draw Seeds Here",
        brush=gr.Brush(
            colors=seed_colors,
            color_mode="fixed",
            default_size=3,
        ),
    )
    proc_btn = gr.Button("Process Seeds")
    info_out = gr.Textbox()
    # state holders
    obj_state = gr.State()          # obj seeds np.ndarray
    bg_state = gr.State()           # bg seeds np.ndarray
    original_state = gr.State()     # original background RGB from the editor

    with gr.Sidebar():
        gr.Markdown(
        """
        ## Instructions
        1. Upload an image using the editor above.
        2. Use the brush tool to draw **Red** seeds on the object and **Blue** seeds on the background.
        3. Click **Process Seeds** to store the seeds.
        4. Adjust the segmentation parameters in the sidebar.
        5. Click **Segment** to run the graph-cut segmentation.
        """
        )
        # slider & algorithm
        lambda_slider = gr.Slider(minimum=0.1, maximum=5.0, value=1.0, step=0.1, label="lambda")
        sigma_slider = gr.Slider(minimum=0.1, maximum=10.0, value=1.0, step=0.1, label="sigma")
        algo_choice = gr.Dropdown(["bk", "bk_nx", "dinic", "edmonds_karp", "preflow"], value="bk", label="algorithm")
        directed_choice = gr.Dropdown(["True", "False"], value="False", label="directed")

    proc_btn.click(process_mask_and_store,
                   inputs=editor,
                   outputs=[info_out, obj_state, bg_state, original_state])

    seg_btn = gr.Button("Segment")
    seg_info = gr.Textbox()
    seg_image = gr.Image(type="numpy")

    seg_btn.click(run_segmentation,
                  inputs=[original_state, obj_state, bg_state, lambda_slider, sigma_slider, algo_choice, directed_choice],
                  outputs=[seg_info, seg_image])


demo.launch()