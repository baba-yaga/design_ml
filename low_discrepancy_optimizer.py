"""
Low-Discrepancy Sequence Simulator and Product Potential Optimizer

This module implements:
1. Generation of low-discrepancy sequences (Sobol', Halton, Latin Hypercube) using scipy.stats.qmc.
2. Optimization of point configurations using a specific product potential energy function.
"""

import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

class LowDiscrepancyGenerator:
    """Wrapper for scipy.stats.qmc to generate scaled low-discrepancy sequences."""
    
    def __init__(self, d, minima=None, maxima=None):
        self.d = d
        self.minima = np.zeros(d) if minima is None else np.array(minima)
        self.maxima = np.ones(d) if maxima is None else np.array(maxima)
        
    def _scale(self, sample):
        """Scale points from [0, 1]^d to [minima, maxima]."""
        return qmc.scale(sample, self.minima, self.maxima)

    def sobol(self, n, scramble=True):
        """Generate Sobol' sequence."""
        sampler = qmc.Sobol(d=self.d, scramble=scramble)
        # Sobol sequence length must be a power of 2 for best properties, 
        # but here we generate standard n
        # Reset is important if reusing, but we create new sampler instance
        sample = sampler.random(n)
        return self._scale(sample)

    def halton(self, n, scramble=True):
        """Generate Halton sequence."""
        sampler = qmc.Halton(d=self.d, scramble=scramble)
        sample = sampler.random(n)
        return self._scale(sample)

    def latin_hypercube(self, n, scramble=True):
        """Generate Latin Hypercube sample."""
        sampler = qmc.LatinHypercube(d=self.d, scramble=scramble)
        sample = sampler.random(n)
        return self._scale(sample)
    
    def random(self, n):
        """Generate independent random points (Poisson process)."""
        sample = np.random.uniform(size=(n, self.d))
        return self._scale(sample)

    def _compute_envelope_bounds(self):
        """
        Compute expanded envelope bounds to ensure coverage after any rotation.
        
        For a d-dimensional box, the envelope is a hypercube centered at the box center
        with side length equal to the diagonal of the original box.
        This ensures that any rotation around the center keeps the original box covered.
        """
        center = (self.minima + self.maxima) / 2
        box_size = self.maxima - self.minima
        
        # Diagonal of the box = sqrt(sum of squared side lengths)
        diagonal = np.sqrt(np.sum(box_size ** 2))
        
        # Envelope is a hypercube with side = diagonal, centered at box center
        envelope_min = center - diagonal / 2
        envelope_max = center + diagonal / 2
        
        return envelope_min, envelope_max
    
    def _random_rotation_matrix(self, angle=None):
        """
        Generate a random rotation matrix for 2D.
        
        For d > 2, this generates a rotation in a random 2D plane.
        
        Parameters:
        -----------
        angle : float or None
            Rotation angle in radians. If None, a random angle in [0, 2π) is used.
        
        Returns:
        --------
        R : ndarray of shape (d, d)
            Rotation matrix
        """
        if angle is None:
            angle = np.random.uniform(0, 2 * np.pi)
        
        if self.d == 2:
            c, s = np.cos(angle), np.sin(angle)
            return np.array([[c, -s], [s, c]])
        elif self.d == 3:
            # Random rotation in 3D using axis-angle representation
            # Generate random axis
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            
            # Rodrigues' rotation formula
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            return R
        else:
            # For d > 3, use a random Givens rotation in a random plane
            R = np.eye(self.d)
            i, j = np.random.choice(self.d, 2, replace=False)
            c, s = np.cos(angle), np.sin(angle)
            R[i, i] = c
            R[j, j] = c
            R[i, j] = -s
            R[j, i] = s
            return R
    
    def _apply_transform(self, points, rotation_matrix, shift_vector, center):
        """
        Apply rotation around center and then shift.
        
        Parameters:
        -----------
        points : ndarray of shape (n, d)
        rotation_matrix : ndarray of shape (d, d)
        shift_vector : ndarray of shape (d,)
        center : ndarray of shape (d,) - rotation center
        
        Returns:
        --------
        transformed : ndarray of shape (n, d)
        """
        # Translate to origin (center becomes 0)
        centered = points - center
        
        # Rotate around origin
        rotated = centered @ rotation_matrix.T
        
        # Translate back and apply shift
        transformed = rotated + center + shift_vector
        
        return transformed
    
    def generate_with_transform(self, n, method='sobol', angle=None, shift=None, 
                                  scramble=True, oversample_factor=2.0, return_info=False):
        """
        Generate points with random rotation and shift.
        
        Points are generated in an expanded envelope region using the specified method,
        transformed (rotated and shifted), and then clipped to the original window.
        
        Parameters:
        -----------
        n : int
            Target number of points (actual count may vary after clipping)
        method : str
            Generation method: 'sobol', 'halton', 'latin_hypercube', or 'random'
        angle : float or None
            Rotation angle in radians. If None, random angle is used.
        shift : array-like or None
            Shift vector. If None, random shift within safe bounds is used.
        scramble : bool
            Whether to use scrambled sequence (for sobol, halton, latin_hypercube)
        oversample_factor : float
            Factor to oversample points (since some will be clipped).
            Larger values give more consistent final point counts.
        return_info : bool
            If True, return dict with transformation info
            
        Returns:
        --------
        points : ndarray of shape (m, d)
            Transformed and clipped points (m ≈ n but may vary)
        info : dict (only if return_info=True)
            Contains 'angle', 'shift', 'rotation_matrix', 'n_generated', 'n_kept', 'method'
        """
        # Compute envelope bounds
        envelope_min, envelope_max = self._compute_envelope_bounds()
        center = (self.minima + self.maxima) / 2
        box_size = self.maxima - self.minima
        
        # Compute how many points to generate (oversample to ensure enough after clip)
        envelope_size = envelope_max - envelope_min
        volume_ratio = np.prod(envelope_size) / np.prod(box_size)
        n_generate = int(n * volume_ratio * oversample_factor)
        n_generate = max(n_generate, n)
        
        # Generate points in envelope using specified method
        if method == 'sobol':
            sampler = qmc.Sobol(d=self.d, scramble=scramble)
            sample_unit = sampler.random(n_generate)
        elif method == 'halton':
            sampler = qmc.Halton(d=self.d, scramble=scramble)
            sample_unit = sampler.random(n_generate)
        elif method == 'latin_hypercube':
            sampler = qmc.LatinHypercube(d=self.d, scramble=scramble)
            sample_unit = sampler.random(n_generate)
        elif method == 'random':
            sample_unit = np.random.uniform(size=(n_generate, self.d))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sobol', 'halton', 'latin_hypercube', or 'random'.")
        
        # Scale to envelope
        sample = qmc.scale(sample_unit, envelope_min, envelope_max)
        
        # Generate rotation matrix
        R = self._random_rotation_matrix(angle)
        actual_angle = angle if angle is not None else np.arctan2(R[1, 0], R[0, 0]) if self.d >= 2 else 0
        
        # Generate shift vector if not provided
        if shift is None:
            # Random shift, bounded so that center doesn't move too far
            max_shift = box_size * 0.1  # 10% of box size
            shift = np.random.uniform(-max_shift, max_shift)
        else:
            shift = np.asarray(shift)
        
        # Apply transformation
        transformed = self._apply_transform(sample, R, shift, center)
        
        # Clip to original window
        mask = np.all((transformed >= self.minima) & (transformed <= self.maxima), axis=1)
        clipped = transformed[mask]
        
        if return_info:
            info = {
                'method': method,
                'angle': actual_angle,
                'shift': shift,
                'rotation_matrix': R,
                'n_generated': n_generate,
                'n_kept': len(clipped),
                'envelope_min': envelope_min,
                'envelope_max': envelope_max
            }
            return clipped, info
        
        return clipped
    
    # Convenience aliases for backward compatibility
    def sobol_with_transform(self, n, **kwargs):
        """Generate Sobol' sequence with random rotation and shift. See generate_with_transform."""
        return self.generate_with_transform(n, method='sobol', **kwargs)
    
    def halton_with_transform(self, n, **kwargs):
        """Generate Halton sequence with random rotation and shift. See generate_with_transform."""
        return self.generate_with_transform(n, method='halton', **kwargs)
    
    def latin_hypercube_with_transform(self, n, **kwargs):
        """Generate Latin Hypercube sample with random rotation and shift. See generate_with_transform."""
        return self.generate_with_transform(n, method='latin_hypercube', **kwargs)
    
    def random_with_transform(self, n, **kwargs):
        """Generate random points with random rotation and shift. See generate_with_transform."""
        return self.generate_with_transform(n, method='random', **kwargs)


class ProductPotentialOptimizer:
    """
    Optimizer for point configurations using a component-wise product potential.
    
    Energy E = Sum_{i != j} p_alpha(x_i, x_j)
    where p_alpha(x, y) = Prod_{k=0}^{d-1} |x[k] - y[k]|^{-alpha}
    """
    
    def __init__(self, d, n, minima=None, maxima=None, alpha=2.0, repulsive_boundary=False, 
                 discount_mode=None, discount_beta=1.0, discount_C=None):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        d : int - Dimension
        n : int - Number of points
        minima : array-like - Box lower bounds
        maxima : array-like - Box upper bounds
        alpha : float - Exponent for potential (default 2.0)
        repulsive_boundary : bool - If True, add boundary repulsion term
        discount_mode : str or None - 'exp', 'power', or 'log' for distance discounting
        discount_beta : float - parameter for 'power' mode
        discount_C : float or None - Custom decay/scale parameter. If None, computed from calibration.
        """
        self.d = d
        self.n = n
        self.minima = np.zeros(d) if minima is None else np.array(minima)
        self.maxima = np.ones(d) if maxima is None else np.array(maxima)
        self.alpha = alpha
        self.repulsive_boundary = repulsive_boundary
        self.discount_mode = discount_mode
        self.discount_beta = discount_beta
        self.discount_C = discount_C
        self.X_optimal = None
        self.optimization_result = None
        
        # Auto-calibrate C if needed
        self._calibrate_C()

    def _calibrate_C(self):
        """Calibrate discount_C if not provided."""
        if self.discount_C is not None:
            return

        if self.discount_mode is None:
            return

        # Define calibration point z_cal = d / 10
        z_cal = self.d / 10.0
        
        if self.discount_mode == 'exp':
             # 0.5 = exp(-C * z_cal) => C = ln(2) / z_cal
             self.discount_C = np.log(2) / z_cal
        elif self.discount_mode == 'power':
             # 0.5 = C * z_cal^{-beta} => C = 0.5 * z_cal^{beta}
             self.discount_C = 0.5 * (z_cal ** self.discount_beta)
        elif self.discount_mode == 'log':
             # 0.5 = C / log(e + z_cal) => C = 0.5 * log(e + z_cal)
             self.discount_C = 0.5 * np.log(np.e + z_cal)

    def _compute_discount_matrix(self, diffs_abs):
        """
        Compute discount factor matrix based on L1 norm distances.
        diffs_abs: (n, n, d) absolute coordinate differences
        """
        if self.discount_mode is None:
            return 1.0
            
        # L1 norm adjusted to box dimensions
        # z[k] = |x[k]-y[k]| / (max[k]-min[k])
        box_size = self.maxima - self.minima
        # Avoid div by zero
        box_size = np.where(box_size < 1e-12, 1.0, box_size)
        
        # Shape: (n, n, d)
        z_k = diffs_abs / box_size[np.newaxis, np.newaxis, :]
        
        # L1 norm ||z||
        z_norm = np.sum(z_k, axis=2)
        
        if self.discount_mode == 'exp':
            # Factor = exp(-C * ||z||)
            return np.exp(-self.discount_C * z_norm)
        
        elif self.discount_mode == 'power':
            # Factor = C * ||z||^{-beta}
            # Add epsilon to z_norm to avoid div by zero
            return self.discount_C * ((z_norm + 1e-10) ** (-self.discount_beta))
            
        elif self.discount_mode == 'log':
            # Factor = C / log(e + ||z||)
            return self.discount_C / np.log(np.e + z_norm)
            
        return 1.0

    def compute_energy(self, X_flat):
        """Compute total energy H(X)."""
        X = X_flat.reshape(self.n, self.d)
        
        # 1. Pairwise Energy
        # We need to compute |x_i[k] - x_j[k]| for all pairs i,j and dim k
        
        # Approach: Expand dims to get diffs matrix (n, n, d)
        # diffs[i, j, k] = x_i[k] - x_j[k]
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        
        # Absolute differences
        abs_diffs = np.abs(diffs)
        
        # Add small epsilon to avoid division by zero during optimization
        # Although if points coincide energy tends to infinity
        epsilon = 1e-10
        
        # Product over dimensions: product_{k} (|diff_ijk| + eps)^(-alpha)
        # = ( product_{k} (|diff_ijk| + eps) )^(-alpha)
        
        # Compute product across last axis (dimensions)
        prod_diffs = np.prod(abs_diffs + epsilon, axis=2)
        
        # Ensure we don't include self-interaction (diagonal is 0)
        np.fill_diagonal(prod_diffs, np.inf) 
        
        # Base Pairwise potentials
        base_potentials = prod_diffs ** (-self.alpha)
        
        # Calculate Discount Factor
        if self.discount_mode:
            discount_factors = self._compute_discount_matrix(abs_diffs)
            potentials_matrix = base_potentials * discount_factors
        else:
            potentials_matrix = base_potentials
        
        # Sum over off-diagonal elements
        energy_pair = np.sum(potentials_matrix[np.isfinite(potentials_matrix)])
        
        energy_boundary = 0.0
        if self.repulsive_boundary:
            # Boundary Term: sum_i prod_k |x_i[k]-min[k]|^{-alpha} * |x_i[k]-max[k]|^{-alpha}
            # = sum_i ( prod_k (|x_i[k]-min[k]| * |x_i[k]-max[k]|) )^{-alpha}
            
            # Dist to min: |x - min|
            d_min = np.abs(X - self.minima)
            # Dist to max: |x - max|
            d_max = np.abs(X - self.maxima)
            
            # Product of distances for each coordinate
            # shape (n, d)
            prod_d = d_min * d_max
            
            # Product over k dimensions
            # shape (n,)
            term_k = np.prod(prod_d + epsilon, axis=1)
            
            # Apply -alpha
            energy_boundary_terms = term_k ** (-self.alpha)
            
            energy_boundary = np.sum(energy_boundary_terms) * self.n
            
        return energy_pair + energy_boundary

    def optimize(self, X_initial=None, maxiter=1000):
        """Minimize energy."""
        if X_initial is None:
            # Start with random points
            X_initial = np.random.uniform(size=(self.n, self.d))
            X_initial = X_initial * (self.maxima - self.minima) + self.minima
            
        bounds = [(self.minima[i % self.d], self.maxima[i % self.d]) 
                  for i in range(self.n * self.d)]
        
        result = minimize(
            self.compute_energy,
            X_initial.flatten(),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )
        
        self.X_optimal = result.x.reshape(self.n, self.d)
        self.optimization_result = result
        return self.X_optimal
