
from low_discrepancy_optimizer import ProductPotentialOptimizer
import numpy as np

def verify_calibration():
    # Test 1: Explicit C
    opt1 = ProductPotentialOptimizer(d=2, n=10, discount_mode='exp', discount_C=5.0)
    print(f"Test 1 (Explicit C=5.0): opt.discount_C = {opt1.discount_C}")
    assert opt1.discount_C == 5.0
    
    # Test 2: Auto-Calibration Exp
    opt2 = ProductPotentialOptimizer(d=2, n=10, discount_mode='exp', discount_C=None)
    # z_cal = 2/10 = 0.2. C = ln(2)/0.2 = ~3.4657
    print(f"Test 2 (Auto Exp): opt.discount_C = {opt2.discount_C}")
    assert opt2.discount_C is not None
    assert np.isclose(opt2.discount_C, np.log(2)/0.2)
    
    # Test 3: Auto-Calibration Power (beta=1.0)
    opt3 = ProductPotentialOptimizer(d=2, n=10, discount_mode='power', discount_beta=1.0, discount_C=None)
    # C = 0.5 * 0.2^1 = 0.1
    print(f"Test 3 (Auto Power): opt.discount_C = {opt3.discount_C}")
    assert opt3.discount_C is not None
    assert np.isclose(opt3.discount_C, 0.1)

    print("All tests passed.")

if __name__ == "__main__":
    verify_calibration()
