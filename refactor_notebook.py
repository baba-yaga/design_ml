
import json
import re

nb_path = '/Users/sergei/articles/design_ml/low_discrepancy_comparison.ipynb'

def update_notebook():
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    # 1. Update analyze_configurations definition
    for cell in cells:
        if cell['cell_type'] == 'code' and 'def analyze_configurations' in ''.join(cell['source']):
            new_source = [
                "def analyze_configurations(d, n, alpha=2.0, repulsive=False, discount_mode=None, discount_beta=1.0, discount_C=None):\n",
                "    print(f\"--- Analyzing Dimension d={d}, Points n={n} ---\\n\")\n",
                "    if discount_mode:\n",
                "        print(f\"    Params: alpha={alpha}, discount_mode={discount_mode}, beta={discount_beta}, C={discount_C}\")\n",
                "    \n",
                "    minima = np.zeros(d)\n",
                "    maxima = np.ones(d)\n",
                "    \n",
                "    # Generators\n",
                "    gen = LowDiscrepancyGenerator(d, minima, maxima)\n",
                "    \n",
                "    configs = {}\n",
                "    \n",
                "    # 1. Random\n",
                "    configs['Random'] = gen.random(n)\n",
                "    \n",
                "    # 2. Sobol\n",
                "    try:\n",
                "        configs['Sobol'] = gen.sobol(n)\n",
                "    except Exception as e:\n",
                "        print(f\"Sobol generation failed (possibly n not power of 2 warning): {e}\")\n",
                "        configs['Sobol'] = np.random.rand(n, d) # Fallback\n",
                "    \n",
                "    # 3. Halton\n",
                "    configs['Halton'] = gen.halton(n)\n",
                "    \n",
                "    # 4. Latin Hypercube\n",
                "    configs['LHS'] = gen.latin_hypercube(n)\n",
                "    \n",
                "    # 5. Optimized (Product Potential)\n",
                "    optimizer = ProductPotentialOptimizer(d, n, minima, maxima, alpha=alpha, repulsive_boundary=repulsive, \n",
                "                                          discount_mode=discount_mode, discount_beta=discount_beta, discount_C=discount_C)\n",
                "    print(\"Optimizing...\")\n",
                "    start_time = time.time()\n",
                "    configs['Optimized'] = optimizer.optimize(maxiter=2000)\n",
                "    print(f\"Optimization done in {time.time() - start_time:.2f}s\")\n",
                "    \n",
                "    # 6. Optimized (Repulsive Boundary)\n",
                "    optimizer_rep = ProductPotentialOptimizer(d, n, minima, maxima, alpha=alpha, repulsive_boundary=True,\n",
                "                                              discount_mode=discount_mode, discount_beta=discount_beta, discount_C=discount_C)\n",
                "    print(\"Optimizing (Repulsive Boundary)...\")\n",
                "    start_time = time.time()\n",
                "    configs['Optimized Repulsive'] = optimizer_rep.optimize(maxiter=2000)\n",
                "    print(f\"Repulsive Optimization done in {time.time() - start_time:.2f}s\")\n",
                "    \n",
                "    # Compute Energies\n",
                "    energies = {}\n",
                "    # For energy comparison, use the same optimizer instance (it holds the energy func)\n",
                "    for name, X in configs.items():\n",
                "        energies[name] = optimizer.compute_energy(X.flatten())\n",
                "        print(f\"{name} Energy: {energies[name]:.4e}\")\n",
                "        \n",
                "    return configs, energies\n"
            ]
            cell['source'] = new_source
            print("Updated analyze_configurations definition.")

    # 2. Update 2D Comparison Cell
    # It usually starts with d = 2, n = 64
    for cell in cells:
        source_str = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'd = 2' in source_str and 'n = 64' in source_str and 'titles =' in source_str:
            new_source = [
                "# --- Simulation Parameters ---\n",
                "# d: Dimension\n",
                "# n: Number of points\n",
                "# alpha: Potential exponent (default 2.0)\n",
                "# discount_mode: 'exp', 'power', 'log', or None\n",
                "# discount_beta: Exponent for power-law discount\n",
                "# discount_C: Decay/scale parameter (None = auto)\n",
                "\n",
                "d = 2\n",
                "n = 64  # Good number for visibility\n",
                "alpha = 2.0\n",
                "discount_mode = None\n",
                "discount_beta = 1.0\n",
                "discount_C = None\n",
                "\n",
                "# -----------------------------\n",
                "\n",
                "configs_2d, energies_2d = analyze_configurations(d, n, alpha=alpha, discount_mode=discount_mode, \n",
                "                                                 discount_beta=discount_beta, discount_C=discount_C)\n",
                "\n",
                "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
                "axes = axes.flatten()\n",
                "\n",
                "titles = ['Random', 'Sobol', 'Halton', 'LHS', 'Optimized', 'Optimized Repulsive']\n",
                "\n",
                "param_str = f\"\\n(α={alpha})\"\n",
                "\n",
                "for i, title in enumerate(titles):\n",
                "    ax = axes[i]\n",
                "    X = configs_2d[title]\n",
                "    E = energies_2d[title]\n",
                "    \n",
                "    ax.scatter(X[:, 0], X[:, 1], s=50, alpha=0.8)\n",
                "    ax.set_title(f\"{title}\\nE = {E:.2e}{param_str}\")\n",
                "    ax.set_xlim(0, 1)\n",
                "    ax.set_ylim(0, 1)\n",
                "    ax.set_aspect('equal')\n",
                "    ax.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            cell['source'] = new_source
            print("Updated 2D Comparison Cell.")

    # 3. Update 1D Comparison Cell
    # d = 1, n = 20
    for cell in cells:
        source_str = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'd = 1' in source_str and 'n = 20' in source_str:
            new_source = [
                "# --- Simulation Parameters ---\n",
                "d = 1\n",
                "n = 20\n",
                "alpha = 2.0\n",
                "discount_mode = None\n",
                "discount_beta = 1.0\n",
                "discount_C = None\n",
                "# -----------------------------\n",
                "\n",
                "configs_1d, energies_1d = analyze_configurations(d, n, alpha=alpha, discount_mode=discount_mode, \n",
                "                                                 discount_beta=discount_beta, discount_C=discount_C)\n",
                "\n",
                "fig, ax = plt.subplots(figsize=(12, 6))\n",
                "\n",
                "y_offsets = range(len(configs_1d))\n",
                "labels = list(configs_1d.keys())\n",
                "\n",
                "param_str = f\" (α={alpha})\"\n",
                "\n",
                "for i, label in enumerate(labels):\n",
                "    X = configs_1d[label]\n",
                "    ax.scatter(X, np.full_like(X, i), s=100, label=label)\n",
                "\n",
                "ax.set_yticks(range(len(labels)))\n",
                "ax.set_yticklabels(labels)\n",
                "ax.set_ylim(-1, len(labels))\n",
                "ax.grid(True, axis='x')\n",
                "ax.set_title(f\"1D Distribution Comparison{param_str}\")\n",
                "plt.show()"
            ]
            cell['source'] = new_source
            print("Updated 1D Comparison Cell.")

    # 4. Update 3D Comparison Cell
    # d = 3, n = 100
    for cell in cells:
        source_str = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and 'd = 3' in source_str and 'n = 100' in source_str:
            new_source = [
                "# --- Simulation Parameters ---\n",
                "d = 3\n",
                "n = 100\n",
                "alpha = 2.0\n",
                "discount_mode = None\n",
                "discount_beta = 1.0\n",
                "discount_C = None\n",
                "# -----------------------------\n",
                "\n",
                "configs_3d, energies_3d = analyze_configurations(d, n, alpha=alpha, discount_mode=discount_mode, \n",
                "                                                 discount_beta=discount_beta, discount_C=discount_C)\n",
                "\n",
                "param_str = f\" (α={alpha})\"\n",
                "\n",
                "for name in ['Random', 'Sobol', 'Optimized']:\n",
                "    X = configs_3d[name]\n",
                "    fig = go.Figure(data=[\n",
                "        go.Scatter3d(\n",
                "            x=X[:,0], y=X[:,1], z=X[:,2],\n",
                "            mode='markers',\n",
                "            marker=dict(size=5, opacity=0.8)\n",
                "        )\n",
                "    ])\n",
                "    fig.update_layout(title=f\"3D - {name} (E={energies_3d[name]:.2e}){param_str}\")\n",
                "    fig.show()"
            ]
            cell['source'] = new_source
            print("Updated 3D Comparison Cell.")

    # 5. Update Discounted Potentials Cell
    # modes = [None, 'exp', 'power', 'log']
    # Loop over modes
    for cell in cells:
        source_str = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and "modes = [None, 'exp', 'power', 'log']" in source_str:
            new_source = [
                "# --- Simulation Parameters ---\n",
                "d = 2\n",
                "n = 64\n",
                "modes = [None, 'exp', 'power', 'log']\n",
                "\n",
                "# Explicit Parameters\n",
                "alpha = 2.0\n",
                "discount_beta = 1.0\n",
                "discount_C = None  # None means auto-calibrated\n",
                "# -----------------------------\n",
                "\n",
                "fig, axes = plt.subplots(2, 2, figsize=(18, 6))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for ax, mode in zip(axes, modes):\n",
                "    print(f\"Optimizing with discount_mode='{mode}', beta={discount_beta}, C={discount_C}...\")\n",
                "    opt = ProductPotentialOptimizer(d, n, \n",
                "                                    repulsive_boundary=True, \n",
                "                                    alpha=alpha,\n",
                "                                    discount_mode=mode, \n",
                "                                    discount_beta=discount_beta,\n",
                "                                    discount_C=discount_C)\n",
                "    X_opt = opt.optimize(maxiter=1500)\n",
                "    E = opt.compute_energy(X_opt.flatten())\n",
                "    \n",
                "    params_str = f\"α={alpha}\"\n",
                "    if mode == 'power':\n",
                "        params_str += f\", β={discount_beta}\"\n",
                "    if opt.discount_C is not None:\n",
                "        params_str += f\", C={opt.discount_C:.2f}\"\n",
                "    \n",
                "    ax.scatter(X_opt[:, 0], X_opt[:, 1], s=50, alpha=0.8)\n",
                "    ax.set_title(f\"Discount: {mode}\\nE={E:.2e}\\n({params_str})\")\n",
                "    ax.set_xlim(0, 1); ax.set_ylim(0, 1)\n",
                "    ax.set_aspect('equal')\n",
                "    ax.grid(True, alpha=0.3)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
            cell['source'] = new_source
            print("Updated Discounted Potentials Comparison Cell.")

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Notebook update complete.")

if __name__ == "__main__":
    update_notebook()
