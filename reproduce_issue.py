import numpy as np
import matplotlib.pyplot as plt
from point_process_optimizer import PointProcessOptimizer

# Set random seed for reproducibility
np.random.seed(42)

def inspect_optimization(kernel, kernel_name, n_points=20, d=2):
    print(f"I: Processing {kernel_name} with {n_points} points...")
    
    optimizer = PointProcessOptimizer(d=d, n=n_points, kernel=kernel, periodic=True)
    X_initial = optimizer.generate_random_points()
    print(f"I: Initial shape: {X_initial.shape}")
    
    optimizer.optimize(X_initial, pin_first_point=True)
    X_opt = optimizer.X_optimal
    print(f"I: Optimal shape: {X_opt.shape}")
    print("I: First 5 points of X_opt:")
    print(X_opt[:5])
    
    # Check for overlaps
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(X_opt)
    print(f"I: Min pairwise distance: {np.min(dists)}")
    print(f"I: Max pairwise distance: {np.max(dists)}")
    
    # Check how many unique points (approx)
    unique_points = np.unique(np.round(X_opt, 4), axis=0)
    print(f"I: Number of unique points (rounded to 4 decimals): {len(unique_points)}")

# Lennard-Jones like kernel
kernel_lj = lambda r: (1/(r+0.1))**12 - 2*(1/(r+0.1))**6
inspect_optimization(kernel_lj, "Lennard-Jones Like")

# Gaussian kernel
kernel_gauss = lambda r: np.exp(-r**2 / 0.5)
inspect_optimization(kernel_gauss, "Gaussian", n_points=20)
