
import json

nb_path = '/Users/sergei/articles/design_ml/low_discrepancy_comparison.ipynb'

def fix_notebook():
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    updated = False
    for cell in cells:
        source_str = ''.join(cell['source'])
        # More robust check for the specific cell
        if cell['cell_type'] == 'code' and "modes = [None, 'log', 'power', 'exp']" in source_str:
            new_source = [
                "# --- Simulation Parameters ---\n",
                "d = 2\n",
                "n = 64\n",
                "modes = [None, 'log', 'power', 'exp']\n",
                "\n",
                "# Explicit Parameters\n",
                "alpha = 2.0\n",
                "discount_beta = 1.0\n",
                "discount_C = None  # None means auto-calibrated\n",
                "# -----------------------------\n",
                "\n",
                "fig, axes = plt.subplots(1, 4, figsize=(18, 6))\n",
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
            updated = True
            print("Updated Discounted Potentials Comparison Cell.")
            break
            
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook saved.")
    else:
        print("Could not find Target Cell to update.")

if __name__ == "__main__":
    fix_notebook()
