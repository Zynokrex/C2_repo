import cv2
import numpy as np
import scipy.sparse as sp
from scipy.stats import gaussian_kde
from graph_solvers import run_maxflow 

class ImageGraph:
    def __init__(self, image: np.ndarray, obj_seeds: np.ndarray, bg_seeds: np.ndarray, directed = False, lambda_val = 1, sigma_val = 1):

        if image.ndim == 3:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = image
        self.directed = directed

        self.obj_seeds = obj_seeds
        self.bg_seeds = bg_seeds
        self.lambda_val = lambda_val
        self.sigma_val = sigma_val

        self.num_pixels = image.shape[0] * image.shape[1]
        self.S_index = self.num_pixels
        self.T_index = self.num_pixels + 1
        self.kde_obj, self.kde_bg = self._compute_bg_obj_kde()

        self.adjacancy_matrix = self._setup_adj_matrix()

    def _setup_adj_matrix(self):
        height, width = self.image.shape
        num_pixels = height * width
        total_nodes = num_pixels + 2  # pixels + S + T
        
        # Initialize lists for sparse matrix construction
        rows, cols, data = [], [], []
        
        max_sum_b = 0
        
        obj_seeds_set = set(map(tuple, self.obj_seeds))
        bg_seeds_set = set(map(tuple, self.bg_seeds))
        
        # Add n-links 
        for y in range(height):
            for x in range(width):
                pixel_idx = y * width + x
                pixel_sum_B = 0
                
                # Check 4-connected neighbors
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_idx = ny * width + nx
                        
                        # Compute boundary penalty
                        intensity_p = float(self.image[y, x])
                        intensity_q = float(self.image[ny, nx])
                        intensity_diff = intensity_p - intensity_q
                        
                        if not self.directed:
                            boundary_cost = np.exp(-(intensity_diff ** 2) / (2 * self.sigma_val ** 2))

                            # Add bidirectional edge (undirected graph)
                            rows.extend([pixel_idx, neighbor_idx])
                            cols.extend([neighbor_idx, pixel_idx])
                            data.extend([boundary_cost, boundary_cost])
                            pixel_sum_B += boundary_cost
                        
                        else: 
                            if intensity_p <= intensity_q:
                                boundary_cost_pq = 1.0
                                boundary_cost_qp = np.exp(-(intensity_diff ** 2) / (2 * self.sigma_val ** 2))
                            else:
                                boundary_cost_pq = np.exp(-(intensity_diff ** 2) / (2 * self.sigma_val ** 2))
                                boundary_cost_qp = 1.0

                            # Add directed edges
                            rows.extend([pixel_idx, neighbor_idx])
                            cols.extend([neighbor_idx, pixel_idx])
                            data.extend([boundary_cost_pq, boundary_cost_qp])

                            # Update sum of max boundary costs
                            pixel_sum_B += max(boundary_cost_pq, boundary_cost_qp)
                                            
                if pixel_sum_B > max_sum_b:
                    max_sum_b = pixel_sum_B
                                    
        K = 1 + max_sum_b
        
        # Add t-links to both S and T
        for y in range(height):
            for x in range(width):
                pixel_idx = y * width + x
                intensity = float(self.image[y, x])
                
                if (x, y) in obj_seeds_set:
                    # Object seed: K to S, 0 to T
                    rows.extend([pixel_idx, self.S_index])
                    cols.extend([self.S_index, pixel_idx])
                    data.extend([K, K]) # {p,S} edges
                    
                    rows.extend([pixel_idx, self.T_index])
                    cols.extend([self.T_index, pixel_idx])
                    data.extend([0, 0]) # {p,T} edges
                    
                elif (x, y) in bg_seeds_set:
                    # Background seed: 0 to S, K to T
                    rows.extend([pixel_idx, self.S_index])
                    cols.extend([self.S_index, pixel_idx])
                    data.extend([0, 0]) # {p,S} edges
                    
                    rows.extend([pixel_idx, self.T_index])
                    cols.extend([self.T_index, pixel_idx])
                    data.extend([K, K]) # {p,T} edges
                    
                else:
                    # Regular pixel: regional costs to both terminals
                    R_bkg = self._compute_regional_cost(intensity, "bg")
                    R_obj = self._compute_regional_cost(intensity, "obj")
                    
                    # {p,S} edges with regional background cost
                    rows.extend([pixel_idx, self.S_index])
                    cols.extend([self.S_index, pixel_idx])
                    data.extend([self.lambda_val * R_bkg, self.lambda_val * R_bkg])
                    
                    # {p,T} edges with regional object cost
                    rows.extend([pixel_idx, self.T_index])
                    cols.extend([self.T_index, pixel_idx])
                    data.extend([self.lambda_val * R_obj, self.lambda_val * R_obj])
        
        adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes))
        
        return adj_matrix

    def _compute_bg_obj_kde(self):
        """
        Compute regional costs using KDE for object and background intensity distributions.
        """
        obj_intensities = np.array([self.image[y, x] for x, y in self.obj_seeds])
        bg_intensities = np.array([self.image[y, x] for x, y in self.bg_seeds])
        
        # Check if we have enough samples
        if len(obj_intensities) < 2:
            raise ValueError(f"Need at least 2 object seed points for KDE. Found {len(obj_intensities)}")
        if len(bg_intensities) < 2:
            raise ValueError(f"Need at least 2 background seed points for KDE. Found {len(bg_intensities)}")
        
        obj_variance = np.var(obj_intensities)
        bg_variance = np.var(bg_intensities)
        
        # Regularization parameter - add small noise if variance is too low
        regularization_noise = 0.1
            
        if obj_variance < 1e-8:
            print("Low object variance detected, adding regularization noise")
            obj_intensities = obj_intensities + np.random.normal(0, regularization_noise, obj_intensities.shape)
        
        if bg_variance < 1e-8:
            print("Low background variance detected, adding regularization noise")
            bg_intensities = bg_intensities + np.random.normal(0, regularization_noise, bg_intensities.shape)
        
        kde_obj = gaussian_kde(obj_intensities)
        kde_bg = gaussian_kde(bg_intensities)
        
        # Store the KDE objects for later use
        self.kde_obj = kde_obj
        self.kde_bg = kde_bg
        self.cost_method = 'kde'
        
        return kde_obj, kde_bg
           
    def _compute_regional_cost(self, intensity, label):
        if label == "obj":
            prob = self.kde_obj.evaluate([intensity])[0]
        elif label == "bg":
            prob = self.kde_bg.evaluate([intensity])[0]
        else:
            raise ValueError('Invalid label when computing regional cost')

        epsilon = 1e-8
        prob = max(prob, epsilon) 
        return -np.log(prob)
    
    def segment(self, algorithm: str):
        """
        Runs graph cut on the built adjacency and returns:
          - flow (float)
          - labels (H x W uint8 mask, 0=foreground/source, 1=background/sink)
        """

        flow, labels_flat = run_maxflow(
            self.adjacancy_matrix, self.S_index, self.T_index, self.num_pixels,
            algorithm=algorithm
        )
        h, w = self.image.shape
        mask = labels_flat.reshape(h, w)
        return flow, mask


