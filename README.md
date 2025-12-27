# Point Process Optimization with Kernel-Based Potentials

This project implements optimization of point configurations in a d-dimensional phase space using pair potentials defined by kernel functions. It provides tools for finding optimal point arrangements that minimize repulsive energy potentials, along with diagnostic functions to analyze spatial point patterns.

## Features

- **Point Configuration Optimization**: Find optimal spatial arrangements of n points in d-dimensional bounded regions
- **Customizable Kernels**: Support for various pair potential functions (default: Coulomb-like 1/r repulsion)
- **Multi-dimensional Support**: Works with 1D, 2D, and 3D configurations
- **Interactive 3D Visualization**: Plotly-based 3D plots with rotating capability and bounding box visualization
- **Spatial Statistics**: Ripley's K-function, Contact distribution F(r), and Nearest-neighbour distribution G(r)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/point-process-optimization.git
cd point-process-optimization

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from point_process_optimizer import PointProcessOptimizer, PointProcessDiagnostics

# Define a repulsive kernel (Coulomb-like)
kernel_repulsive = lambda r: 1.0 / (r + 1e-8)

# Create optimizer for 2D with 20 points in unit square
opt_2d = PointProcessOptimizer(d=2, n=20, kernel=kernel_repulsive)

# Optimize configuration
X_opt = opt_2d.optimize()

# Visualize result
opt_2d.plot_configuration(title='Optimized 2D Configuration')
```

### 3D Example with Custom Domain

```python
# 3D optimization in a rectangular parallelepiped
opt_3d = PointProcessOptimizer(
    d=3, 
    n=30,
    minima=[0, 0, 0],
    maxima=[2, 1, 1],
    kernel=kernel_repulsive
)

X_opt_3d = opt_3d.optimize()
opt_3d.plot_configuration(title='Optimized 3D Configuration')
```

### Diagnostic Analysis

```python
# Analyze point pattern with diagnostic functions
diagnostics = PointProcessDiagnostics(X_opt, minima=[0, 0], maxima=[1, 1])

# Plot K-function, Contact distribution, and Nearest-neighbour distribution
diagnostics.plot_all_diagnostics()
```

### Low-Discrepancy Sequences with Random Transform

Generate quasi-Monte Carlo sequences (Sobol', Halton, Latin Hypercube) with random rotation and shift:

```python
from low_discrepancy_optimizer import LowDiscrepancyGenerator

gen = LowDiscrepancyGenerator(d=2, minima=[0, 0], maxima=[1, 1])

# Generate Sobol' points with random rotation and shift
points = gen.sobol_with_transform(n=100)

# Or use the generic method with any QMC sequence
points = gen.generate_with_transform(n=100, method='halton')

# Get transformation details
points, info = gen.sobol_with_transform(n=100, return_info=True)
print(f"Angle: {np.degrees(info['angle'])}°, Kept: {info['n_kept']} points")
```

## API Reference

### `PointProcessOptimizer`

#### Constructor
```python
PointProcessOptimizer(d, n, minima=None, maxima=None, kernel=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | int | Dimension of the phase space |
| `n` | int | Number of points |
| `minima` | array-like | Minimum values for each dimension (default: 0) |
| `maxima` | array-like | Maximum values for each dimension (default: 1) |
| `kernel` | callable | Kernel function K(r) (default: 1/r repulsive) |

#### Methods

- `generate_random_points()`: Generate n random uniform points in S
- `compute_potential(X_flat)`: Compute the total potential H(X)
- `optimize(X_initial=None, method='L-BFGS-B', maxiter=1000)`: Find optimal configuration
- `plot_configuration(X=None, title='Point Configuration')`: Visualize the configuration

### `PointProcessDiagnostics`

#### Constructor
```python
PointProcessDiagnostics(X, minima=None, maxima=None)
```

#### Methods

- `ripley_k_function(r_values=None, n_r=50)`: Compute Ripley's K-function
- `contact_distribution(r_values=None, n_samples=1000, n_r=50)`: Compute contact distribution F(r)
- `nearest_neighbour_distribution(r_values=None, n_r=50)`: Compute nearest-neighbour distribution G(r)
- `plot_all_diagnostics()`: Plot all diagnostic functions

### `LowDiscrepancyGenerator`

#### Constructor
```python
LowDiscrepancyGenerator(d, minima=None, maxima=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `d` | int | Dimension |
| `minima` | array-like | Box lower bounds (default: 0) |
| `maxima` | array-like | Box upper bounds (default: 1) |

#### Methods

- `sobol(n, scramble=True)`: Generate Sobol' sequence
- `halton(n, scramble=True)`: Generate Halton sequence
- `latin_hypercube(n, scramble=True)`: Generate Latin Hypercube sample
- `random(n)`: Generate random uniform points
- `generate_with_transform(n, method='sobol', angle=None, shift=None, ...)`: Generate QMC points with random rotation and shift

#### `generate_with_transform` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | int | required | Target number of points |
| `method` | str | 'sobol' | 'sobol', 'halton', 'latin_hypercube', or 'random' |
| `angle` | float/None | None | Rotation angle in radians (None = random) |
| `shift` | array/None | None | Shift vector (None = random) |
| `oversample_factor` | float | 2.0 | Oversampling factor |
| `return_info` | bool | False | Return transformation info dict |

## Mathematical Background

The optimizer minimizes the energy potential:

$$H(X) = \sum_{i < j} K(||x_i - x_j||)$$

where $K(r)$ is a kernel function defining pair interactions. The default kernel $K(r) = 1/r$ creates Coulomb-like repulsion, causing points to spread out and fill the space uniformly.

### Diagnostic Functions

- **Ripley's K-function**: Measures the expected number of points within distance r of a typical point
- **Contact Distribution F(r)**: CDF of distance from a random location to the nearest point
- **Nearest-neighbour Distribution G(r)**: CDF of distance from each point to its nearest neighbour

## Project Structure

```
point-process-optimization/
├── point_process_optimizer.py    # Main module with optimizer and diagnostics classes
├── low_discrepancy_optimizer.py  # Low-discrepancy sequences and product potential optimizer
├── point_process_optimization.ipynb  # Interactive Jupyter notebook examples
├── qmc_transform_demo.ipynb      # QMC sequences with random rotation/shift demo
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

- Python 3.9+
- NumPy
- SciPy
- Matplotlib
- Plotly
- Jupyter (for notebooks)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
