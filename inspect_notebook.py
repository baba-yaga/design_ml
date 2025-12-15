import json

with open('/Users/sergei/articles/design_ml/pp_transformations.ipynb', 'r') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "Lennard" in source or "optimize_and_visualize" in source:
            print(f"Cell {i} Source:")
            print(source)
            print("-" * 20)
