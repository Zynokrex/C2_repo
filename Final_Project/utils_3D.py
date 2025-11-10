import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def create_synthetic_volume(size: int = 64,
                            line_intensity: int = 200,
                            line_radius: int = 2):
    """
    Create a 3D volume (D=H=W=size) with a single 3D spiral of constant radius.

    Returns:
        volume: uint8 array of shape (size, size, size) with values in [0, 255].
        lines_voxels: list with a single list of voxel coordinates for the spiral (z, y, x),
                      kept as a list-of-lists for compatibility with existing code.
    """
    D = H = W = size
    volume = np.zeros((D, H, W), dtype=np.uint8)

    # For compatibility with the rest of the code (expects lines_voxels[0])
    lines_voxels = [[]]

    cx = cy = size * 0.5

    # Spiral parameters (constant radius, no growth/waviness)
    n_turns = 2
    base_radius = size * 0.18  # fixed radius

    for z in range(D):
        t = z / (D - 1 + 1e-8)          # normalized height in [0,1]
        theta = 2.0 * np.pi * n_turns * t

        # Constant radius spiral around the center
        x_center = cx + base_radius * np.cos(theta)
        y_center = cy + base_radius * np.sin(theta)

        xc = int(round(x_center))
        yc = int(round(y_center))
        zz = z

        if 0 <= zz < D:
            for dy in range(-line_radius, line_radius + 1):
                for dx in range(-line_radius, line_radius + 1):
                    yy = yc + dy
                    xx = xc + dx
                    if 0 <= yy < H and 0 <= xx < W:
                        volume[zz, yy, xx] = line_intensity
                        lines_voxels[0].append((zz, yy, xx))

    return volume, lines_voxels

def create_seeds_from_middle_slice(volume: np.ndarray,
                                   lines_voxels,
                                   n_obj_seeds: int = 15,
                                   n_bg_seeds: int = 40):
    """
    Use only the middle slice as in the paper:
    - Object seeds: subset of voxels belonging to the spiral on that slice.
    - Background seeds: random voxels from the same slice where intensity == 0.
    """
    D, H, W = volume.shape
    z_mid = D // 2

    # All voxels of the spiral on the middle slice
    line0_on_mid = [(z, y, x) for (z, y, x) in lines_voxels[0] if z == z_mid]
    if len(line0_on_mid) < 2:
        raise RuntimeError("Not enough voxels of the spiral on the middle slice to seed from")

    # Sample object seeds
    idxs = np.linspace(0, len(line0_on_mid) - 1,
                       num=min(n_obj_seeds, len(line0_on_mid)),
                       dtype=int)
    obj_voxels = [line0_on_mid[i] for i in idxs]

    # Convert to (x, y, z)
    obj_seeds = np.array([[x, y, z] for (z, y, x) in obj_voxels],
                         dtype=np.int64)

    # Background seeds from middle slice where volume is 0
    zeros_mask = (volume[z_mid] == 0)
    ys, xs = np.where(zeros_mask)
    if ys.size < 2:
        raise RuntimeError("Not enough background voxels on the middle slice")
    perm = np.random.permutation(ys.size)
    bg_count = min(n_bg_seeds, ys.size)
    ys_sel = ys[perm[:bg_count]]
    xs_sel = xs[perm[:bg_count]]
    z_arr = np.full(bg_count, z_mid, dtype=np.int64)
    bg_seeds = np.stack([xs_sel, ys_sel, z_arr], axis=1)

    return obj_seeds, bg_seeds, z_mid

def save_visualizations_2d(volume: np.ndarray,
                           obj_seeds: np.ndarray,
                           bg_seeds: np.ndarray,
                           z_mid: int,
                           out_dir: Path):
    """
    Save 2D visualizations for the middle slice.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_img = volume[z_mid]
    slice_bgr = cv2.cvtColor(slice_img, cv2.COLOR_GRAY2BGR)

    seeds_vis = slice_bgr.copy()
    for x, y, z in obj_seeds:
        if z == z_mid and 0 <= y < slice_img.shape[0] and 0 <= x < slice_img.shape[1]:
            seeds_vis[y, x] = (0, 0, 255)  # red
    for x, y, z in bg_seeds:
        if z == z_mid and 0 <= y < slice_img.shape[0] and 0 <= x < slice_img.shape[1]:
            seeds_vis[y, x] = (255, 0, 0)  # blue

    cv2.imwrite(str(out_dir / "slice_middle.png"), slice_bgr)
    cv2.imwrite(str(out_dir / "slice_middle_with_seeds.png"), seeds_vis)

def _plot_voxels(filled: np.ndarray, colors: np.ndarray, title: str = None):
    """
    Helper: make a voxel plot with equal aspect (cube).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(filled, facecolors=colors, edgecolor='k')

    # Equal aspect: use data shape
    sx, sy, sz = filled.shape
    ax.set_box_aspect((sx, sy, sz))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title is not None:
        ax.set_title(title)
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    return fig, ax

def _scatter_seeds_3d(ax, obj_seeds: np.ndarray, bg_seeds: np.ndarray, vol_shape):
    """
    Overlay object and background seeds as 3D points on the same coordinate system
    used by ax.voxels.

    volume has shape (D, H, W) == (z, y, x)
    but seeds are stored as (x, y, z), so we convert to (z, y, x).
    """
    D, H, W = vol_shape

    if obj_seeds.size > 0:
        # (x, y, z) -> (z, y, x) to match voxel indexing
        oz = obj_seeds[:, 2]
        oy = obj_seeds[:, 1]
        ox = obj_seeds[:, 0]
        # keep only in-bounds
        mask = (oz >= 0) & (oz < D) & (oy >= 0) & (oy < H) & (ox >= 0) & (ox < W)
        ax.scatter(oz[mask], oy[mask], ox[mask], c='red', s=20, depthshade=False, label='obj seeds')

    if bg_seeds.size > 0:
        bz = bg_seeds[:, 2]
        by = bg_seeds[:, 1]
        bx = bg_seeds[:, 0]
        mask = (bz >= 0) & (bz < D) & (by >= 0) & (by < H) & (bx >= 0) & (bx < W)
        ax.scatter(bz[mask], by[mask], bx[mask], c='blue', s=15, depthshade=False, label='bg seeds')

    # optional legend (small)
    ax.legend(loc='upper right', fontsize=8)

def save_3d_volume_view(volume: np.ndarray,
                        obj_seeds: np.ndarray,
                        bg_seeds: np.ndarray,
                        out_path: Path):
    """
    3D visualization of the synthetic object + seeds.
    Black voxels (0) are transparent. Non-zero voxels are white-ish cubes.
    """
    filled = volume > 0
    if not np.any(filled):
        print("No non-zero voxels to visualize in 3D.")
        return

    colors = np.zeros(filled.shape + (4,), dtype=float)
    colors[filled] = (1.0, 1.0, 1.0, 0.8)

    fig, ax = _plot_voxels(filled, colors, title="Original volume + seeds")
    _scatter_seeds_3d(ax, obj_seeds, bg_seeds, volume.shape)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def save_3d_segmented_view(volume: np.ndarray,
                           mask_3d: np.ndarray,
                           obj_seeds: np.ndarray,
                           bg_seeds: np.ndarray,
                           out_path: Path):
    """
    3D visualization of the segmented volume + seeds.
    Only voxels where volume > 0 are shown.
    Object (mask==0) in RED, non-object (mask==1) in light gray.
    """
    filled = volume > 0
    if not np.any(filled):
        print("No non-zero voxels to visualize in 3D segmentation.")
        return

    obj = (mask_3d == 0) & filled
    bg = (mask_3d == 1) & filled

    colors = np.zeros(filled.shape + (4,), dtype=float)
    # Object voxels: RED
    colors[obj] = (1.0, 0.0, 0.0, 0.9)
    # Non-object but non-zero: light gray, semi-transparent
    colors[bg] = (0.7, 0.7, 0.7, 0.4)

    fig, ax = _plot_voxels(filled, colors, title="Segmented volume (object in red) + seeds")
    _scatter_seeds_3d(ax, obj_seeds, bg_seeds, volume.shape)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def show_3d_volume_view_interactive(volume: np.ndarray,
                                    obj_seeds: np.ndarray,
                                    bg_seeds: np.ndarray):
    """
    Interactive 3D view of the original volume + seeds.
    """
    filled = volume > 0
    if not np.any(filled):
        print("No non-zero voxels to visualize in 3D.")
        return

    colors = np.zeros(filled.shape + (4,), dtype=float)
    colors[filled] = (1.0, 1.0, 1.0, 0.8)

    fig, ax = _plot_voxels(filled, colors, title="Original volume + seeds (interactive)")
    _scatter_seeds_3d(ax, obj_seeds, bg_seeds, volume.shape)
    plt.show()

def show_3d_segmented_view_interactive(volume: np.ndarray,
                                       mask_3d: np.ndarray,
                                       obj_seeds: np.ndarray,
                                       bg_seeds: np.ndarray):
    """
    Interactive 3D view of the segmented volume + seeds.
    """
    filled = volume > 0
    if not np.any(filled):
        print("No non-zero voxels to visualize in 3D segmentation.")
        return

    obj = (mask_3d == 0) & filled
    bg = (mask_3d == 1) & filled

    colors = np.zeros(filled.shape + (4,), dtype=float)
    colors[obj] = (1.0, 0.0, 0.0, 0.9)
    colors[bg] = (0.7, 0.7, 0.7, 0.4)

    fig, ax = _plot_voxels(filled, colors, title="Segmented volume + seeds (interactive)")
    _scatter_seeds_3d(ax, obj_seeds, bg_seeds, volume.shape)
    plt.show()