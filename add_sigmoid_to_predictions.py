#!/usr/bin/env python3
"""
Add sigmoid to prediction/inference lines in training cells.
Handles patterns like: model(X).cpu().numpy().flatten()
"""

import json
import re
from pathlib import Path

NOTEBOOKS = [
    "experiments/MC1_experiment.ipynb",
    "experiments/MC2_experiment.ipynb",
    "experiments/MW1_experiment.ipynb",
    "experiments/PC1_experiment.ipynb",
    "experiments/PC2_experiment.ipynb",
    "experiments/PC3_experiment.ipynb",
    "experiments/PC4_experiment.ipynb",
]

def add_sigmoid_to_predictions(source_code):
    """Add torch.sigmoid to model prediction lines"""
    
    # Skip if already has sigmoid
    if source_code.count('torch.sigmoid') > 2:  # Allow some, but not all
        return source_code
    
    lines = source_code.split('\n')
    updated_lines = []
    
    for line in lines:
        updated_line = line
        
        # Pattern: variable = model_name(X_tensor).cpu().numpy()...
        # Match patterns like:
        # - y_val_proba_kan = model_kan(X_val_tensor).cpu().numpy().flatten()
        # - y_test_proba_kan = model_kan(X_test_tensor).cpu().numpy().flatten()
        # - val_outputs = model_kan(X_val_tensor).cpu().numpy().flatten()
        # - predictions = model(X).cpu().numpy()
        
        # Look for model calls followed by .cpu() or .numpy()
        # that are NOT already wrapped in torch.sigmoid
        if 'model' in line and '(' in line and '.cpu()' in line or '.numpy()' in line:
            # Check if NOT in training loop (has .eval() context or with torch.no_grad)
            # and NOT already has sigmoid
            if 'torch.sigmoid' not in line and '= model' in line or '= self.model' in line:
                # Pattern: var = model_xxx(X_xxx).cpu()...
                match = re.search(r'(\s*\w+\s*=\s*)(model\w*)\(([^)]+)\)(\.cpu\(\)\.numpy\(\)\.flatten\(\)|\.cpu\(\)\.numpy\(\)|\.squeeze\(\))', line)
                if match:
                    indent = match.group(1)
                    model_name = match.group(2)
                    input_tensor = match.group(3)
                    chain = match.group(4)
                    updated_line = f"{indent}torch.sigmoid({model_name}({input_tensor})){chain}"
                    print(f"    Updated: {line.strip()[:60]}...")
        
        updated_lines.append(updated_line)
    
    return '\n'.join(updated_lines)

def update_training_cells(notebook_path):
    """Update training cells (22, 24) with sigmoid predictions"""
    print(f"\nProcessing: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook.get('cells', [])
    updated = False
    
    # Update cell-22
    if len(cells) > 22:
        cell_22 = cells[22]
        if cell_22.get('cell_type') == 'code':
            original = ''.join(cell_22['source'])
            if 'BCEWithLogitsLoss' in original:
                print("  Cell 22 (KAN training):")
                updated_source = add_sigmoid_to_predictions(original)
                if updated_source != original:
                    cell_22['source'] = updated_source.splitlines(keepends=True)
                    updated = True
                else:
                    print("    No changes needed")
    
    # Update cell-24
    if len(cells) > 24:
        cell_24 = cells[24]
        if cell_24.get('cell_type') == 'code':
            original = ''.join(cell_24['source'])
            if 'BCEWithLogitsLoss' in original:
                print("  Cell 24 (KAN_Attention training):")
                updated_source = add_sigmoid_to_predictions(original)
                if updated_source != original:
                    cell_24['source'] = updated_source.splitlines(keepends=True)
                    updated = True
                else:
                    print("    No changes needed")
    
    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print("  ✓ Saved updates")
    
    return updated

def main():
    print("=" * 70)
    print("Adding sigmoid to predictions in training cells")
    print("=" * 70)
    
    base_path = Path("/home/user/nasa-defect-gwo-kan")
    updated_count = 0
    
    for notebook_file in NOTEBOOKS:
        notebook_path = base_path / notebook_file
        if notebook_path.exists():
            if update_training_cells(notebook_path):
                updated_count += 1
        else:
            print(f"\n⚠ Not found: {notebook_path}")
    
    print("\n" + "=" * 70)
    print(f"✓ Updated {updated_count}/{len(NOTEBOOKS)} notebooks")
    print("=" * 70)

if __name__ == "__main__":
    main()
