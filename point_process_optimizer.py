"""
Point Process Optimization with Kernel-Based Potentials

This module implements optimization of point configurations in a d-dimensional 
phase space using pair potentials defined by a kernel function.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform
import plotly.graph_objects as go


class PointProcessOptimizer:
    """Optimizer for point configurations using kernel-based pair potentials."""
    
    def __init__(self, d, n, minima=None, maxima=None, kernel=None, periodic=True):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        d : int - Dimension of the phase space
        n : int - Number of points
        minima : array-like - Minimum values for each dimension (default: 0)
        maxima : array-like - Maximum values for each dimension (default: 1)
        kernel : callable - Kernel function K(r) (default: 1/r repulsive)
        periodic : bool - If True, use periodic boundary conditions (torus topology)
        """
        self.d = d
        self.n = n
        self.minima = np.zeros(d) if minima is None else np.array(minima)
        self.maxima = np.ones(d) if maxima is None else np.array(maxima)
        self.box_size = self.maxima - self.minima  # Size of box in each dimension
        self.kernel = kernel if kernel else lambda r: 1.0 / (r + 1e-10)
        self.periodic = periodic
        self.X_optimal = None
    
    def generate_random_points(self):
        """Generate n random uniform points in S."""
        X = np.random.uniform(size=(self.n, self.d))
        return X * (self.maxima - self.minima) + self.minima
    
    def _compute_pairwise_distances(self, X):
        """Compute pairwise distances, with minimum image convention if periodic."""
        if not self.periodic:
            return pdist(X)
        
        # Vectorized minimum image convention
        # Diff matrix: shape (n, n, d)
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        
        # Wrap differences to [-L/2, L/2] bounds (minimum image)
        diffs -= self.box_size * np.round(diffs / self.box_size)
        
        # Euclidean norms
        dists_matrix = np.linalg.norm(diffs, axis=-1)
        
        # Return condensed distance matrix to match pdist output (upper triangle)
        return squareform(dists_matrix, checks=False)
    
    def _wrap_to_box(self, X):
        """Wrap points back into the box [minima, maxima] using modular arithmetic."""
        return self.minima + np.mod(X - self.minima, self.box_size)
    
    def _compute_potential_from_config(self, X):
        """Compute potential from a configuration array X of shape (n, d)."""
        distances = self._compute_pairwise_distances(X)
        potentials = self.kernel(distances)
        return np.sum(potentials)

    def compute_potential(self, X_flat):
        """Compute the total potential H(X)."""
        X = X_flat.reshape(self.n, self.d)
        if self.periodic:
            X = self._wrap_to_box(X)  # Ensure points are in canonical box
        return self._compute_potential_from_config(X)
    
    def optimize(self, X_initial=None, method='L-BFGS-B', maxiter=1000, pin_first_point=None):
        """Find configuration that minimizes potential.
        
        Parameters:
        -----------
        X_initial : array-like, optional - Initial configuration (default: random)
        method : str - Optimization method (default: 'L-BFGS-B' for bounded, 'BFGS' for periodic)
        maxiter : int - Maximum iterations (default: 1000)
        pin_first_point : bool, optional - If True, fix first point to break translational
            symmetry. Default is True for periodic, False otherwise.
        """
        if X_initial is None:
            X_initial = self.generate_random_points()
        
        # Default: pin first point only for periodic (to break translational symmetry)
        if pin_first_point is None:
            pin_first_point = self.periodic
        
        if self.periodic:
            if pin_first_point:
                # Pin first point to break translational symmetry
                # This dramatically improves convergence by eliminating the valley of
                # equivalent solutions that differ only by a global translation
                pinned_point = X_initial[0:1].copy()  # Shape (1, d)
                X_free_initial = X_initial[1:]  # Shape (n-1, d)
                
                def objective_pinned(X_flat):
                    X_free = X_flat.reshape(self.n - 1, self.d)
                    X_full = np.vstack([pinned_point, self._wrap_to_box(X_free)])
                    return self._compute_potential_from_config(X_full)
                
                result = minimize(
                    objective_pinned,
                    X_free_initial.flatten(),
                    method='BFGS',
                    options={'maxiter': maxiter, 'disp': False}
                )
                # Reconstruct full configuration
                X_free_opt = result.x.reshape(self.n - 1, self.d)
                self.X_optimal = np.vstack([pinned_point, self._wrap_to_box(X_free_opt)])
            else:
                # No pinning - may converge slowly due to translation invariance
                result = minimize(
                    self.compute_potential,
                    X_initial.flatten(),
                    method='BFGS',
                    options={'maxiter': maxiter, 'disp': False}
                )
                self.X_optimal = self._wrap_to_box(result.x.reshape(self.n, self.d))
        else:
            # Set bounds for non-periodic case
            bounds = [(self.minima[i % self.d], self.maxima[i % self.d]) 
                      for i in range(self.n * self.d)]
            result = minimize(
                self.compute_potential,
                X_initial.flatten(),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': False}
            )
            self.X_optimal = result.x.reshape(self.n, self.d)
        
        self.optimization_result = result
        return self.X_optimal
    
    def plot_configuration(self, X=None, title='Point Configuration'):
        """Plot the point configuration."""
        if X is None:
            X = self.X_optimal
        if X is None:
            raise ValueError("No configuration to plot. Run optimize() first.")
        
        if self.d == 1:
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.scatter(X[:, 0], np.zeros(self.n), s=100, c='royalblue', edgecolors='navy')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlim(self.minima[0] - 0.1, self.maxima[0] + 0.1)
            ax.set_ylim(-0.5, 0.5)
            ax.set_xlabel('x', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_yticks([])
            plt.tight_layout()
            return fig
            
        elif self.d == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(X[:, 0], X[:, 1], s=100, c='royalblue', edgecolors='navy')
            ax.set_xlim(self.minima[0] - 0.1, self.maxima[0] + 0.1)
            ax.set_ylim(self.minima[1] - 0.1, self.maxima[1] + 0.1)
            ax.set_xlabel('x₁', fontsize=12)
            ax.set_ylabel('x₂', fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
            
        elif self.d == 3:
            # Use Plotly for interactive 3D
            # Create edges of the parallelepiped S
            x0, y0, z0 = self.minima
            x1, y1, z1 = self.maxima
            # Define all 12 edges of the parallelepiped as line segments
            edges_x = [x0, x1, None, x0, x1, None, x0, x1, None, x0, x1, None,  # edges along x
                       x0, x0, None, x1, x1, None, x0, x0, None, x1, x1, None,  # edges along y
                       x0, x0, None, x1, x1, None, x0, x0, None, x1, x1, None]  # edges along z
            edges_y = [y0, y0, None, y1, y1, None, y0, y0, None, y1, y1, None,
                       y0, y1, None, y0, y1, None, y0, y1, None, y0, y1, None,
                       y0, y0, None, y0, y0, None, y1, y1, None, y1, y1, None]
            edges_z = [z0, z0, None, z0, z0, None, z1, z1, None, z1, z1, None,
                       z0, z0, None, z0, z0, None, z1, z1, None, z1, z1, None,
                       z0, z1, None, z0, z1, None, z0, z1, None, z0, z1, None]
            
            fig = go.Figure(data=[
                # Parallelepiped edges in grey
                go.Scatter3d(
                    x=edges_x, y=edges_y, z=edges_z,
                    mode='lines',
                    line=dict(color='grey', width=2),
                    name='Boundary S',
                    showlegend=False
                ),
                # Points
                go.Scatter3d(
                    x=X[:, 0], y=X[:, 1], z=X[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='royalblue',
                        line=dict(color='navy', width=1)
                    ),
                    name='Points'
                )
            ])
            fig.update_layout(
                title=f'{title} (drag to rotate)',
                scene=dict(
                    xaxis_title='x₁',
                    yaxis_title='x₂',
                    zaxis_title='x₃'
                ),
                width=700,
                height=600
            )
            fig.show()
            return fig
        else:
            print(f"Cannot plot {self.d}-dimensional configuration directly.")
            return None

    def random_shift(self, X, magnitude=None):
        """
        Apply a random shift to the configuration X.
        
        Parameters:
        -----------
        X : array-like - Point configuration
        magnitude : float, optional - Scaling factor for shift (default: n^(-1/d))
        """
        if magnitude is None:
            magnitude = self.n ** (-1.0 / self.d)
            
        # v_i uniformly distributed in [0.5*magnitude*(minima-maxima), 0.5*magnitude*(maxima-minima)]
        # This creates a random shift vector v within a box scaled by magnitude
        ranges = self.maxima - self.minima
        low = 0.5 * magnitude * (self.minima - self.maxima)
        high = 0.5 * magnitude * (self.maxima - self.minima)
        
        shift_vector = np.random.uniform(low, high, size=(1, self.d))
        X_shifted = X + shift_vector
        
        if self.periodic:
            return self._wrap_to_box(X_shifted)
        else:
            return np.clip(X_shifted, self.minima, self.maxima)

    def random_rotation(self, X, center=None, continuous_rotation=False):
        """
        Apply a random rotation/reflection to the configuration X.
        
        If periodic=True and continuous_rotation=False, samples from the hyperoctahedral 
        group B_d (signed permutations) to preserve toroidal distances.
        
        If periodic=False or continuous_rotation=True, samples uniformly from O(d).
        Note: Continuous rotation on a torus breaks the distance metric visually if projected
        to a single box, but is useful for visualizing configuration symmetries/randomness.
        
        Parameters:
        -----------
        X : array-like - Point configuration
        center : array-like, optional - Center of rotation (default: center of box)
        continuous_rotation : bool, optional - If True, force continuous rotation even if periodic.
        """
        if center is None:
            center = (self.minima + self.maxima) / 2.0
            
        if self.periodic and not continuous_rotation:
            # For periodic boundaries (torus), continuous rotations break the distance metric.
            # We must sample from the symmetry group of the lattice (Hyperoctahedral group B_d).
            # This consists of signed permutation matrices.
            
            # 1. Random permutation of axes
            perm = np.random.permutation(self.d)
            
            # 2. Random signs (reflections)
            signs = np.random.choice([-1, 1], size=self.d)
            
            # Apply transformation: x'_i = s_i * (x_{p_i} - c_{p_i}) + c_i
            # Actually, simpler to view as: center coordinate system, permute/reflect, wrap.
            
            X_centered = X - center
            X_new = np.zeros_like(X)
            
            for i in range(self.d):
                X_new[:, i] = signs[i] * X_centered[:, perm[i]]
                
            X_rotated = X_new + center
            
            return self._wrap_to_box(X_rotated)
            
        else:
            # Non-periodic OR forced continuous: continuous rotation in O(d)
            if self.d == 1:
                # O(1) is {-1, 1}. Randomly choose to reflect or not.
                if np.random.rand() < 0.5:
                    # Reflection around center
                    X_rotated = 2*center - X
                else:
                    return X.copy()
                
                # Handle boundaries
                if self.periodic:
                    return self._wrap_to_box(X_rotated)
                else:
                    return np.clip(X_rotated, self.minima, self.maxima)
            
            # Generate random rotation matrix using QR decomposition of Gaussian matrix
            H = np.random.randn(self.d, self.d)
            Q, R = np.linalg.qr(H)
            
            X_centered = X - center
            X_rotated = X_centered @ Q.T + center
            
            if self.periodic:
                return self._wrap_to_box(X_rotated)
            else:
                return np.clip(X_rotated, self.minima, self.maxima)

    def plot_random_transformations(self, X=None, k=3, title='Random Shift + Rotations'):
        """
        Plot original configuration and k random shift+rotations.
        
        Parameters:
        -----------
        X : array-like, optional - Base configuration (default: optimal)
        k : int - Number of random transformations to generate
        title : str - Plot title
        """
        if X is None:
            X = self.X_optimal
        if X is None:
            raise ValueError("No configuration provided.")
            
        # Generate k transformed configurations
        transformed_configs = []
        for _ in range(k):
            # Apply both rotation and shift
            X_new = self.random_rotation(X)
            X_new = self.random_shift(X_new)
            transformed_configs.append(X_new)
            
        if self.d == 1:
            fig, ax = plt.subplots(figsize=(12, 4))
            # Original
            ax.scatter(X[:, 0], np.zeros(self.n), s=100, c='black', label='Original', zorder=10)
            
            # Transformed
            for i, Xt in enumerate(transformed_configs):
                # Random color not too dark, not too light
                color = np.random.uniform(0.3, 0.9, 3)
                ax.scatter(Xt[:, 0], np.ones(self.n) * (i + 1), s=50, color=color, label=f'Trans {i+1}')
                
            ax.set_xlim(self.minima[0] - 0.1, self.maxima[0] + 0.1)
            ax.set_yticks(range(k + 1))
            ax.set_yticklabels(['Orig'] + [f'T{i+1}' for i in range(k)])
            ax.set_xlabel('x')
            ax.set_title(title)
            ax.legend(loc='upper right')
            plt.tight_layout()
            return fig
            
        elif self.d == 2:
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Transformed
            for i, Xt in enumerate(transformed_configs):
                color = np.random.uniform(0.3, 0.9, 3)
                ax.scatter(Xt[:, 0], Xt[:, 1], s=50, color=color, alpha=0.6, label=f'Trans {i+1}')
                
            # Original on top
            ax.scatter(X[:, 0], X[:, 1], s=100, c='black', edgecolors='white', label='Original', zorder=10)
            
            ax.set_xlim(self.minima[0] - 0.1, self.maxima[0] + 0.1)
            ax.set_ylim(self.minima[1] - 0.1, self.maxima[1] + 0.1)
            ax.set_xlabel('x₁')
            ax.set_ylabel('x₂')
            ax.set_title(title)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()
            return fig
            
        elif self.d == 3:
            # Setup similar to plot_configuration
            x0, y0, z0 = self.minima
            x1, y1, z1 = self.maxima
            edges_x = [x0, x1, None, x0, x1, None, x0, x1, None, x0, x1, None,
                       x0, x0, None, x1, x1, None, x0, x0, None, x1, x1, None,
                       x0, x0, None, x1, x1, None, x0, x0, None, x1, x1, None]
            edges_y = [y0, y0, None, y1, y1, None, y0, y0, None, y1, y1, None,
                       y0, y1, None, y0, y1, None, y0, y1, None, y0, y1, None,
                       y0, y0, None, y0, y0, None, y1, y1, None, y1, y1, None]
            edges_z = [z0, z0, None, z0, z0, None, z1, z1, None, z1, z1, None,
                       z0, z0, None, z0, z0, None, z1, z1, None, z1, z1, None,
                       z0, z1, None, z0, z1, None, z0, z1, None, z0, z1, None]
            
            data = [
                go.Scatter3d(
                    x=edges_x, y=edges_y, z=edges_z,
                    mode='lines',
                    line=dict(color='grey', width=2),
                    name='Boundary S',
                    showlegend=False
                )
            ]
            
            # Transformed points
            for i, Xt in enumerate(transformed_configs):
                # Random color
                color_rgb = np.random.uniform(50, 230, 3).astype(int)
                color_hex = f'rgb({color_rgb[0]},{color_rgb[1]},{color_rgb[2]})'
                
                data.append(go.Scatter3d(
                    x=Xt[:, 0], y=Xt[:, 1], z=Xt[:, 2],
                    mode='markers',
                    marker=dict(size=6, color=color_hex, opacity=0.6),
                    name=f'Trans {i+1}'
                ))
            
            # Original points
            data.append(go.Scatter3d(
                x=X[:, 0], y=X[:, 1], z=X[:, 2],
                mode='markers',
                marker=dict(size=8, color='black', line=dict(color='white', width=1)),
                name='Original'
            ))
            
            fig = go.Figure(data=data)
            fig.update_layout(
                title=f'{title} (drag to rotate)',
                scene=dict(xaxis_title='x₁', yaxis_title='x₂', zaxis_title='x₃'),
                width=700, height=600
            )
            fig.show()
            return fig
            
        else:
            print(f"Cannot plot {self.d}-dimensional configuration.")
            return None


class PointProcessDiagnostics:
    """Diagnostic functions for point configurations."""
    
    def __init__(self, X, minima=None, maxima=None, periodic=False):
        self.X = np.array(X)
        self.n, self.d = self.X.shape
        self.minima = np.zeros(self.d) if minima is None else np.array(minima)
        self.maxima = np.ones(self.d) if maxima is None else np.array(maxima)
        self.box_size = self.maxima - self.minima
        self.periodic = periodic
        self.volume = np.prod(self.box_size)
        self.intensity = self.n / self.volume
    
    def _pairwise_distance_matrix(self, X1, X2=None):
        """Compute pairwise distances with minimum image convention if periodic."""
        if X2 is None:
            X2 = X1
            same = True
        else:
            same = False
        
        if not self.periodic:
            if same:
                return squareform(pdist(X1))
            else:
                return cdist(X1, X2)
        
        # Minimum image convention
        n1, n2 = len(X1), len(X2)
        distances = np.zeros((n1, n2))
        for i in range(n1):
            delta = X2 - X1[i]
            delta = delta - self.box_size * np.round(delta / self.box_size)
            distances[i] = np.linalg.norm(delta, axis=1)
        return distances
    
    def ripley_k_function(self, r_values=None, n_r=50):
        """Compute Ripley's K-function."""
        if r_values is None:
            max_r = np.min(self.maxima - self.minima) / 2
            r_values = np.linspace(0, max_r, n_r)
        
        distances = self._pairwise_distance_matrix(self.X)
        K_values = np.zeros(len(r_values))
        
        for i, r in enumerate(r_values):
            count = np.sum(distances < r) - self.n  # Exclude diagonal
            K_values[i] = (self.volume / (self.n * (self.n - 1))) * count
        
        return r_values, K_values
    
    def contact_distribution(self, r_values=None, n_samples=1000, n_r=50):
        """Compute contact distribution function F(r)."""
        if r_values is None:
            max_r = np.min(self.maxima - self.minima) / 4
            r_values = np.linspace(0, max_r, n_r)
        
        # Generate random sample points
        samples = np.random.uniform(size=(n_samples, self.d))
        samples = samples * (self.maxima - self.minima) + self.minima
        
        # Min distance from each sample to configuration points
        min_distances = self._pairwise_distance_matrix(samples, self.X).min(axis=1)
        
        F_values = np.array([np.mean(min_distances <= r) for r in r_values])
        return r_values, F_values
    
    def nearest_neighbour_distribution(self, r_values=None, n_r=50):
        """Compute nearest-neighbour distribution G(r): 
        empirical CDF of distances from each point to its nearest neighbour."""
        if r_values is None:
            max_r = np.min(self.maxima - self.minima) / 4
            r_values = np.linspace(0, max_r, n_r)
    
        # Compute pairwise distances
        distances = self._pairwise_distance_matrix(self.X)
        # Set diagonal to infinity to exclude self-distances
        np.fill_diagonal(distances, np.inf)
        # Get nearest neighbour distance for each point
        nn_distances = distances.min(axis=1)
        # Compute empirical CDF
        G_values = np.array([np.mean(nn_distances <= r) for r in r_values])
        return r_values, G_values
    
    def plot_all_diagnostics(self):
        """Plot all diagnostic functions."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # K-function
        r, K = self.ripley_k_function()
        axes[0].plot(r, K, 'b-', linewidth=2, label='K(r)')
        if self.d == 2:
            axes[0].plot(r, np.pi * r**2, 'r--', label='πr² (Poisson)')
        axes[0].set_xlabel('r')
        axes[0].set_ylabel('K(r)')
        axes[0].set_title("Ripley's K-function")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Contact distribution
        r, F = self.contact_distribution()
        axes[1].plot(r, F, 'b-', linewidth=2, label='F(r)')
        axes[1].set_xlabel('r')
        axes[1].set_ylabel('F(r)')
        axes[1].set_title('Contact Distribution F(r)')
        axes[1].grid(True, alpha=0.3)

        # Nearest-neighbour distribution
        r, G = self.nearest_neighbour_distribution()
        axes[2].plot(r, G, 'b-', linewidth=2, label='G(r)')
        if self.d == 2:
            # Theoretical G for Poisson: G(r) = 1 - exp(-λπr²)
            axes[2].plot(r, 1 - np.exp(-self.intensity * np.pi * r**2), 'r--', label='Poisson') 
            axes[2].set_xlabel('r')
            axes[2].set_ylabel('G(r)')
            axes[2].set_title('Nearest-Neighbour Distribution G(r)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
