
import time
import numpy as np
from point_process_optimizer import PointProcessOptimizer

def verify_performance():
    print("Verifying Pinned Point Optimization Performance (nfev check)...")
    
    # Setup
    d = 2
    n = 20
    np.random.seed(42)
    
    # Initialize optimizer
    opt = PointProcessOptimizer(d=d, n=n, periodic=True)
    X_init = opt.generate_random_points()
    
    # Test 1: Without pinning
    print("\nRunning WITHOUT pinning...")
    start_time = time.time()
    opt.optimize(X_initial=X_init.copy(), maxiter=1000, pin_first_point=False)
    duration_unpinned = time.time() - start_time
    nfev_unpinned = opt.optimization_result.nfev
    print(f"Time: {duration_unpinned:.4f}s, nfev: {nfev_unpinned}")
    
    # Test 2: With pinning
    print("\nRunning WITH pinning...")
    start_time = time.time()
    opt.optimize(X_initial=X_init.copy(), maxiter=1000, pin_first_point=True)
    duration_pinned = time.time() - start_time
    nfev_pinned = opt.optimization_result.nfev
    print(f"Time: {duration_pinned:.4f}s, nfev: {nfev_pinned}")
    
    # Comparison
    print(f"\nTime Speedup: {duration_unpinned / duration_pinned:.2f}x")
    print(f"Nfev Reduction: {nfev_unpinned / nfev_pinned:.2f}x")

if __name__ == "__main__":
    verify_performance()
