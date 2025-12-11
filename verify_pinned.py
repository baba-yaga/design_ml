
import time
import numpy as np
from point_process_optimizer import PointProcessOptimizer

def verify_performance():
    print("Verifying Pinned Point Optimization Performance...")
    
    # Setup
    d = 2
    n = 30
    np.random.seed(42)
    
    # Initialize optimizer
    opt = PointProcessOptimizer(d=d, n=n, periodic=True)
    X_init = opt.generate_random_points()
    
    # Test 1: Without pinning (Old behavior)
    print("\nRunning WITHOUT pinning (may be slow)...")
    start_time = time.time()
    X_opt_unpinned = opt.optimize(X_initial=X_init.copy(), maxiter=1000, pin_first_point=False)
    duration_unpinned = time.time() - start_time
    print(f"Time without pinning: {duration_unpinned:.4f} seconds")
    
    # Test 2: With pinning (New behavior)
    print("\nRunning WITH pinning (New default behavior)...")
    start_time = time.time()
    X_opt_pinned = opt.optimize(X_initial=X_init.copy(), maxiter=1000, pin_first_point=True)
    duration_pinned = time.time() - start_time
    print(f"Time with pinning: {duration_pinned:.4f} seconds")
    
    # Calculate speedup
    speedup = duration_unpinned / duration_pinned
    print(f"\nSpeedup factor: {speedup:.2f}x")
    
    if speedup > 2.0:
        print("SUCCESS: Significant speedup observed.")
    else:
        print("WARNING: Speedup is less than expected.")

if __name__ == "__main__":
    verify_performance()
