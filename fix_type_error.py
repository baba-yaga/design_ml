
import json

nb_path = '/Users/sergei/articles/design_ml/low_discrepancy_comparison.ipynb'

def fix_notebook():
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    
    updated = False
    for cell in cells:
        source_str = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and "discount_C = [None, None, None, None, 1.0, 5.0]" in source_str:
            new_source = [
                "# --- Simulation Parameters ---\n",
                "d = 2\n",
                "n = 64\n",
                "modes = [None, 'log', 'power', 'exp', 'exp', 'exp']\n",
                "\n",
                "# Explicit Parameters\n",
                "alpha = 2.0\n",
                "discount_beta = 1.0\n",
                "discount_C_list = [None, None, None, None, 1.0, 5.0]  # Renamed to avoid confusion\n",
                "# -----------------------------\n",
                "\n",
                "fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n",
                "axes = axes.flatten()\n",
                "\n",
                "for ax, mode, C_val in zip(axes, modes, discount_C_list):\n",
                "    print(f\"Optimizing with discount_mode='{mode}', beta={discount_beta}, C={C_val}...\")\n",
                "    opt = ProductPotentialOptimizer(d, n, \n",
                "                                    repulsive_boundary=True, \n",
                "                                    alpha=alpha,\n",
                "                                    discount_mode=mode, \n",
                "                                    discount_beta=discount_beta,\n",
                "                                    discount_C=C_val)\n",
                "    X_opt = opt.optimize(maxiter=1500)\n",
                "    E = opt.compute_energy(X_opt.flatten())\n",
                "    \n",
                "    params_str = f\"α={alpha}\"\n",
                "    if mode == 'power':\n",
                "        params_str += f\", β={discount_beta}\"\n",
                "    # Now opt.discount_C is guaranteed to be the float used (auto or explicit)\n",
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
            print("Updated loop to zip over C values.")
            break
            
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook saved.")
    else:
        print("Could not find failing cell to update.")

if __name__ == "__main__":
    fix_notebook()
