import json

notebook_path = '/Users/sergei/articles/design_ml/pp_transformations.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "kernel_gauss = lambda r: np.exp(-r**2 / 0.5)" in source:
            print("Found target cell. Update kernel width...")
            new_source = source.replace("0.5", "0.05")
            # Determine if source is list of strings or single string in JSON (usually list in file, but load might make it list)
            # json.load keeps it as list of strings usually if that's how it is on disk.
            # But here I joined it. Let's check type in `cell['source']`
            
            # Actually, let's just replace in the list directly to be safe about preserving format
            new_lines = []
            for line in cell['source']:
                new_lines.append(line.replace("0.5", "0.05"))
            cell['source'] = new_lines
            print("Updated cell source.")

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
