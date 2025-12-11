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
    
    def __init__(self, d, n, minima=None, maxima=None, kernel=None, periodic=False):
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
        
        # Minimum image convention for periodic boundaries
        n = X.shape[0]
        distances = []
        for i in range(n):
            for j in range(i + 1, n):
                delta = X[i] - X[j]
                # Wrap differences to [-L/2, L/2] for each dimension
                delta = delta - self.box_size * np.round(delta / self.box_size)
                distances.append(np.linalg.norm(delta))
        return np.array(distances)
    
    def _wrap_to_box(self, X):
        """Wrap points back into the box [minima, maxima] using modular arithmetic."""
        return self.minima + np.mod(X - self.minima, self.box_size)
    
    def compute_potential(self, X_flat):
        """Compute the total potential H(X)."""
        X = X_flat.reshape(self.n, self.d)
        if self.periodic:
            X = self._wrap_to_box(X)  # Ensure points are in canonical box
        distances = self._compute_pairwise_distances(X)
        potentials = self.kernel(distances)
        return np.sum(potentials)
    
    def optimize(self, X_initial=None, method='L-BFGS-B', maxiter=1000):
        """Find configuration that minimizes potential."""
        if X_initial is None:
            X_initial = self.generate_random_points()
        
        if self.periodic:
            # No bounds needed - topology handles wrapping
            # Use a method that doesn't require bounds
            result = minimize(
                self.compute_potential,
                X_initial.flatten(),
                method='BFGS',
                options={'maxiter': maxiter, 'disp': False}
            )
            # Wrap final result back into canonical box
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
