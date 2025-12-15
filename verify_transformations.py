
import numpy as np
import matplotlib.pyplot as plt
from point_process_optimizer import PointProcessOptimizer

def verify_transformations():
    print("Verifying transformations...")
    
    # Test 2D case
    print("\n--- 2D Case ---")
    opt_2d = PointProcessOptimizer(d=2, n=10, periodic=True)
    X = opt_2d.generate_random_points()
    opt_2d.X_optimal = X
    
    print("Testing random_shift...")
    X_shifted = opt_2d.random_shift(X)
    print(f"Shifted shape: {X_shifted.shape}")
    
    print("Testing random_rotation...")
    X_rotated = opt_2d.random_rotation(X)
    print(f"Rotated shape: {X_rotated.shape}")
    
    print("Testing plot_random_transformations (2D)...")
    try:
        fig = opt_2d.plot_random_transformations(k=3)
        print("Plot 2D generated successfully.")
        plt.close(fig)
    except Exception as e:
        print(f"Plot 2D failed: {e}")

    # Test 3D case
    print("\n--- 3D Case ---")
    opt_3d = PointProcessOptimizer(d=3, n=10, periodic=False)
    X = opt_3d.generate_random_points()
    opt_3d.X_optimal = X
    
    print("Testing random_shift...")
    X_shifted = opt_3d.random_shift(X)
    
    print("Testing random_rotation...")
    X_rotated = opt_3d.random_rotation(X)
    
    print("Testing plot_random_transformations (3D)...")
    try:
        # Note: plotly show() might try to open a browser, which we can't see but code execution shouldn't crash.
        # However, in a headless environment, it might just print or do nothing.
        # We just want to ensure no exceptions are raised during figure construction.
        fig = opt_3d.plot_random_transformations(k=2)
        print("Plot 3D generated successfully.")
    except Exception as e:
        print(f"Plot 3D failed: {e}")

if __name__ == "__main__":
    verify_transformations()
