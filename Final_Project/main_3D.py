import numpy as np
import scipy.sparse as sp
from scipy.stats import gaussian_kde
from pathlib import Path
from datetime import datetime
import numpy as np
from pathlib import Path

from graph_solvers import run_maxflow
from utils_3D import create_synthetic_volume, create_seeds_from_middle_slice
from utils_3D import save_3d_volume_view, save_3d_segmented_view, save_visualizations_2d
from utils_3D import show_3d_volume_view_interactive, show_3d_segmented_view_interactive

class VolumeGraph:
    def __init__(self, volume: np.ndarray,
                 obj_seeds: np.ndarray,
                 bg_seeds: np.ndarray,
                 lambda_val: float = 1.0,
                 sigma_val: float = 1.0) -> None:
        """
        3D analogue of the 2D ImageGraph.

        Args:
            volume: 3D array (D, H, W) or (Z, Y, X) with grayscale intensities.
            obj_seeds: array of shape (N_obj, 3) with (x, y, z) integer coordinates.
            bg_seeds: array of shape (N_bg, 3) with (x, y, z) integer coordinates.
        """
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
        self.volume = volume.astype(np.float32)
        self.obj_seeds = np.asarray(obj_seeds, dtype=np.int64)
        self.bg_seeds = np.asarray(bg_seeds, dtype=np.int64)
        self.lambda_val = float(lambda_val)
        self.sigma_val = float(sigma_val)

        self.D, self.H, self.W = self.volume.shape
        self.num_voxels = int(self.D * self.H * self.W)
        self.S_index = self.num_voxels
        self.T_index = self.num_voxels + 1

        self.kde_obj, self.kde_bg = self._compute_bg_obj_kde()
        self.adjacency_matrix = self._setup_adj_matrix()

    def _voxel_index(self, z: int, y: int, x: int) -> int:
        """Flatten (z, y, x) into a single node index."""
        return int(z * self.H * self.W + y * self.W + x)

    def _compute_bg_obj_kde(self):
        """
        Same logic as in the 2D version, but using 3D seeds.
        """
        if self.obj_seeds.size == 0:
            raise ValueError("No object seeds provided")
        if self.bg_seeds.size == 0:
            raise ValueError("No background seeds provided")

        obj_intensities = []
        for x, y, z in self.obj_seeds:
            if 0 <= z < self.D and 0 <= y < self.H and 0 <= x < self.W:
                obj_intensities.append(self.volume[z, y, x])
        bg_intensities = []
        for x, y, z in self.bg_seeds:
            if 0 <= z < self.D and 0 <= y < self.H and 0 <= x < self.W:
                bg_intensities.append(self.volume[z, y, x])

        obj_intensities = np.asarray(obj_intensities, dtype=np.float64)
        bg_intensities = np.asarray(bg_intensities, dtype=np.float64)

        if obj_intensities.size < 2:
            raise ValueError(f"Need at least 2 object seed voxels for KDE, got {obj_intensities.size}")
        if bg_intensities.size < 2:
            raise ValueError(f"Need at least 2 background seed voxels for KDE, got {bg_intensities.size}")

        obj_var = np.var(obj_intensities)
        bg_var = np.var(bg_intensities)
        regularization_noise = 0.1

        if obj_var < 1e-8:
            print("Low object variance detected (3D), adding regularization noise")
            obj_intensities = obj_intensities + np.random.normal(0, regularization_noise, obj_intensities.shape)

        if bg_var < 1e-8:
            print("Low background variance detected (3D), adding regularization noise")
            bg_intensities = bg_intensities + np.random.normal(0, regularization_noise, bg_intensities.shape)

        kde_obj = gaussian_kde(obj_intensities)
        kde_bg = gaussian_kde(bg_intensities)
        self.cost_method = "kde"

        return kde_obj, kde_bg

    def _compute_regional_cost(self, intensity: float, label: str) -> float:
        if label == "obj":
            prob = float(self.kde_obj.evaluate([intensity])[0])
        elif label == "bg":
            prob = float(self.kde_bg.evaluate([intensity])[0])
        else:
            raise ValueError("Invalid label when computing regional cost")

        epsilon = 1e-8
        prob = max(prob, epsilon)
        return -np.log(prob)

    def _setup_adj_matrix(self) -> sp.csr_matrix:
        """
        Build adjacency for a 3D 6-connected grid.

        Nodes: all voxels + 2 terminals (S, T).
        """
        D, H, W = self.D, self.H, self.W
        num_voxels = self.num_voxels
        total_nodes = num_voxels + 2

        rows = []
        cols = []
        data = []

        max_sum_b = 0.0

        # Convert seeds to sets of tuples for fast lookup
        obj_seeds_set = {tuple(coord) for coord in self.obj_seeds}
        bg_seeds_set = {tuple(coord) for coord in self.bg_seeds}

        # N-links (6-connected)
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    p_idx = self._voxel_index(z, y, x)
                    voxel_sum_B = 0.0

                    for dz, dy, dx in [(-1, 0, 0), (1, 0, 0),
                                       (0, -1, 0), (0, 1, 0),
                                       (0, 0, -1), (0, 0, 1)]:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < D and 0 <= ny < H and 0 <= nx < W:
                            q_idx = self._voxel_index(nz, ny, nx)
                            intensity_diff = float(self.volume[z, y, x] - self.volume[nz, ny, nx])
                            boundary_cost = np.exp(-(intensity_diff ** 2) / (2.0 * (self.sigma_val ** 2)))

                            # bidirectional edge
                            rows.extend([p_idx, q_idx])
                            cols.extend([q_idx, p_idx])
                            data.extend([boundary_cost, boundary_cost])

                            voxel_sum_B += boundary_cost

                    if voxel_sum_B > max_sum_b:
                        max_sum_b = voxel_sum_B

        K = 1.0 + max_sum_b

        # T-links (regional terms and hard constraints)
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    p_idx = self._voxel_index(z, y, x)
                    intensity = float(self.volume[z, y, x])
                    key = (x, y, z)

                    if key in obj_seeds_set:
                        # Object seed: hard constraint to SOURCE
                        rows.extend([p_idx, self.S_index])
                        cols.extend([self.S_index, p_idx])
                        data.extend([K, K])

                        rows.extend([p_idx, self.T_index])
                        cols.extend([self.T_index, p_idx])
                        data.extend([0.0, 0.0])
                    elif key in bg_seeds_set:
                        # Background seed: hard constraint to SINK
                        rows.extend([p_idx, self.S_index])
                        cols.extend([self.S_index, p_idx])
                        data.extend([0.0, 0.0])

                        rows.extend([p_idx, self.T_index])
                        cols.extend([self.T_index, p_idx])
                        data.extend([K, K])
                    else:
                        R_bkg = self._compute_regional_cost(intensity, "bg")
                        R_obj = self._compute_regional_cost(intensity, "obj")

                        rows.extend([p_idx, self.S_index])
                        cols.extend([self.S_index, p_idx])
                        data.extend([self.lambda_val * R_bkg, self.lambda_val * R_bkg])

                        rows.extend([p_idx, self.T_index])
                        cols.extend([self.T_index, p_idx])
                        data.extend([self.lambda_val * R_obj, self.lambda_val * R_obj])

        adj = sp.csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
        return adj

    def segment(self, algorithm: str = "bk"):
        """
        Run s/t min-cut on the 3D adjacency.

        Returns:
            flow: max-flow value.
            mask: 3D uint8 array (D, H, W), 0=SOURCE (object), 1=SINK (background).
        """
        flow, labels_flat = run_maxflow(
            self.adjacency_matrix, self.S_index, self.T_index, self.num_voxels, algorithm=algorithm
        )
        mask = labels_flat.reshape(self.D, self.H, self.W)
        return flow, mask

def main():
    np.random.seed(0)

    # Create synthetic volume
    volume, lines_voxels = create_synthetic_volume(size=35)

    # Create seeds from middle slice
    obj_seeds, bg_seeds, z_mid = create_seeds_from_middle_slice(volume, lines_voxels, n_obj_seeds=3, n_bg_seeds=5)

    print(f"Volume shape: {volume.shape}")
    print(f"Number of object seeds: {len(obj_seeds)}")
    print(f"Number of background seeds: {len(bg_seeds)}")
    print(f"Middle slice index used for seeding: {z_mid}")

    # Build volume graph
    vg = VolumeGraph(volume, obj_seeds, bg_seeds, lambda_val=1.0, sigma_val=5.0)

    # Segment volume
    print("Running 3D graph cut...")
    _, mask_3d = vg.segment(algorithm="bk")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("./results_3d") / f"exp_3d_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save visualizations
    save_visualizations_2d(volume, obj_seeds, bg_seeds, z_mid, out_dir)
    save_3d_volume_view(volume, obj_seeds, bg_seeds, out_dir / "volume_3d_original_with_seeds.png")
    save_3d_segmented_view(volume, mask_3d, obj_seeds, bg_seeds, out_dir / "volume_3d_segmented_with_seeds.png")
    show_3d_volume_view_interactive(volume, obj_seeds, bg_seeds)
    show_3d_segmented_view_interactive(volume, mask_3d, obj_seeds, bg_seeds)

if __name__ == "__main__":
    main()
