
# ... (Previous PointProcessOptimizer class definition) ...
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, cdist, squareform

class PointProcessOptimizer:
    """Optimizer for point configurations using kernel-based pair potentials."""
    
    def __init__(self, d, n, minima=None, maxima=None, kernel=None, periodic=True):
        self.d = d
        self.n = n
        self.minima = np.zeros(d) if minima is None else np.array(minima)
        self.maxima = np.ones(d) if maxima is None else np.array(maxima)
        self.box_size = self.maxima - self.minima
        self.kernel = kernel if kernel else lambda r: 1.0 / (r + 1e-10)
        self.periodic = periodic
        self.X_optimal = None
    
    def generate_random_points(self):
        X = np.random.uniform(size=(self.n, self.d))
        return X * (self.maxima - self.minima) + self.minima
    
    def _compute_pairwise_distances(self, X):
        if not self.periodic:
            return pdist(X)
        
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        diffs -= self.box_size * np.round(diffs / self.box_size)
        dists_matrix = np.linalg.norm(diffs, axis=-1)
        return squareform(dists_matrix, checks=False)
    
    def _wrap_to_box(self, X):
        return self.minima + np.mod(X - self.minima, self.box_size)
    
    def _compute_potential_from_config(self, X):
        distances = self._compute_pairwise_distances(X)
        potentials = self.kernel(distances)
        return np.sum(potentials)

    def compute_potential(self, X_flat):
        X = X_flat.reshape(self.n, self.d)
        if self.periodic:
            X = self._wrap_to_box(X)
        return self._compute_potential_from_config(X)
    
    def optimize(self, X_initial=None, method='L-BFGS-B', maxiter=1000, pin_first_point=None):
        if X_initial is None:
            X_initial = self.generate_random_points()
        
        if pin_first_point is None:
            pin_first_point = self.periodic
        
        if self.periodic:
            if pin_first_point:
                pinned_point = X_initial[0:1].copy()
                X_free_initial = X_initial[1:]
                
                def objective_pinned(X_flat):
                    X_free = X_flat.reshape(self.n - 1, self.d)
                    X_full = np.vstack([pinned_point, self._wrap_to_box(X_free)])
                    return self._compute_potential_from_config(X_full)
                
                result = minimize(
                    objective_pinned,
                    X_free_initial.flatten(),
                    # Use method='L-BFGS-B' to avoid potentially unstable BFGS updates if any?
                    # The original code uses 'BFGS' for periodic. Let's stick to it.
                    method='BFGS',
                    options={'maxiter': maxiter, 'disp': False}
                )
                X_free_opt = result.x.reshape(self.n - 1, self.d)
                self.X_optimal = np.vstack([pinned_point, self._wrap_to_box(X_free_opt)])
            else:
                result = minimize(
                    self.compute_potential,
                    X_initial.flatten(),
                    method='BFGS',
                    options={'maxiter': maxiter, 'disp': False}
                )
                self.X_optimal = self._wrap_to_box(result.x.reshape(self.n, self.d))
        else:
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

# --- Inspection Script ---

np.random.seed(42)

def inspect_optimization(kernel, kernel_name, n_points=20, d=2):
    print(f"I: Processing {kernel_name} with {n_points} points...")
    optimizer = PointProcessOptimizer(d=d, n=n_points, kernel=kernel, periodic=True)
    X_initial = optimizer.generate_random_points()
    optimizer.optimize(X_initial, pin_first_point=True)
    X_opt = optimizer.X_optimal
    unique_points = np.unique(np.round(X_opt, 3), axis=0)
    print(f"I: Number of unique points: {len(unique_points)}")
    if len(unique_points) < n_points:
        print("I: COLLAPSE DETECTED! Unique points:")
        print(unique_points)

# Narrow Gaussian
print("Testing Narrow Gaussian (width 0.05)")
kernel_gauss_narrow = lambda r: np.exp(-r**2 / 0.05)
inspect_optimization(kernel_gauss_narrow, "Gaussian Narrow")

# Original Gaussian
print("\nTesting Original Gaussian (width 0.5)")
kernel_gauss = lambda r: np.exp(-r**2 / 0.5)
inspect_optimization(kernel_gauss, "Gaussian Original")
