#!/usr/bin/env python3
"""
Batch update NASA defect experiment notebooks with hyperparameter optimizations.
Updates cells 10, 11, 22, and 24 in MC1, MC2, MW1, PC1, PC2, PC3, PC4 notebooks.
"""

import json
import re
from pathlib import Path

# Notebook files to update
NOTEBOOKS = [
    "experiments/MC1_experiment.ipynb",
    "experiments/MC2_experiment.ipynb",
    "experiments/MW1_experiment.ipynb",
    "experiments/PC1_experiment.ipynb",
    "experiments/PC2_experiment.ipynb",
    "experiments/PC3_experiment.ipynb",
    "experiments/PC4_experiment.ipynb",
]

def update_kan_model(source_code):
    """Update KAN model: grid_size=3 -> 10, remove sigmoid, return x"""
    # Change grid_size=3 to grid_size=10
    source_code = re.sub(r'grid_size\s*=\s*3', 'grid_size=10', source_code)
    
    # Replace torch.sigmoid(x) with x in return statement
    source_code = re.sub(
        r'return\s+torch\.sigmoid\(x\)',
        'return x',
        source_code
    )
    
    return source_code

def update_training_cell(source_code):
    """Update training cell: Add pos_weight, BCEWithLogitsLoss, sigmoid during inference"""
    
    # Check if pos_weight already exists
    if 'pos_weight = torch.tensor([3.5])' in source_code:
        return source_code
    
    # Find the criterion line and replace it
    # Look for: criterion = nn.BCELoss()
    source_code = re.sub(
        r'criterion\s*=\s*nn\.BCELoss\(\)',
        'pos_weight = torch.tensor([3.5]).to(device)\n    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)',
        source_code
    )
    
    # Add sigmoid during inference/evaluation
    # Find prediction lines like: y_pred = model(X_batch)
    # Replace with: y_pred = torch.sigmoid(model(X_batch))
    
    # For validation loop predictions
    source_code = re.sub(
        r'(\s+)val_outputs\s*=\s*model\(X_val_batch\)(\s*\.squeeze\(\))?',
        r'\1val_outputs = torch.sigmoid(model(X_val_batch))\2',
        source_code
    )
    
    # For test predictions (after training)
    # Look for: y_pred = model(X_test_tensor).squeeze()
    source_code = re.sub(
        r'(\s+)y_pred\s*=\s*model\((X_test_tensor|X_test)\)(\.squeeze\(\))?',
        r'\1y_pred = torch.sigmoid(model(\2))\3',
        source_code
    )
    
    # Also handle cases where it's split across lines
    # predictions = model(X_test_tensor)
    source_code = re.sub(
        r'(\s+)predictions\s*=\s*model\((X_test_tensor|X_test)\)',
        r'\1predictions = torch.sigmoid(model(\2))',
        source_code
    )
    
    return source_code

def update_notebook(notebook_path):
    """Update a single notebook with all changes"""
    print(f"\nProcessing: {notebook_path}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    cells = notebook.get('cells', [])
    updated_cells = 0
    
    # Update cell-10 (KAN model)
    if len(cells) > 10:
        cell_10 = cells[10]
        if cell_10.get('cell_type') == 'code':
            original_source = ''.join(cell_10['source'])
            if 'class KAN' in original_source and 'grid_size' in original_source:
                updated_source = update_kan_model(original_source)
                cell_10['source'] = updated_source.splitlines(keepends=True)
                updated_cells += 1
                print(f"  ✓ Updated cell-10 (KAN model)")
    
    # Update cell-11 (KAN_Attention model)
    if len(cells) > 11:
        cell_11 = cells[11]
        if cell_11.get('cell_type') == 'code':
            original_source = ''.join(cell_11['source'])
            if 'class KAN_Attention' in original_source and 'grid_size' in original_source:
                updated_source = update_kan_model(original_source)
                cell_11['source'] = updated_source.splitlines(keepends=True)
                updated_cells += 1
                print(f"  ✓ Updated cell-11 (KAN_Attention model)")
    
    # Update cell-22 (KAN training)
    if len(cells) > 22:
        cell_22 = cells[22]
        if cell_22.get('cell_type') == 'code':
            original_source = ''.join(cell_22['source'])
            if 'criterion' in original_source and 'nn.BCELoss' in original_source:
                updated_source = update_training_cell(original_source)
                cell_22['source'] = updated_source.splitlines(keepends=True)
                updated_cells += 1
                print(f"  ✓ Updated cell-22 (KAN training)")
    
    # Update cell-24 (KAN_Attention training)
    if len(cells) > 24:
        cell_24 = cells[24]
        if cell_24.get('cell_type') == 'code':
            original_source = ''.join(cell_24['source'])
            if 'criterion' in original_source and 'nn.BCELoss' in original_source:
                updated_source = update_training_cell(original_source)
                cell_24['source'] = updated_source.splitlines(keepends=True)
                updated_cells += 1
                print(f"  ✓ Updated cell-24 (KAN_Attention training)")
    
    # Save updated notebook
    if updated_cells > 0:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  → Saved {updated_cells} cell updates")
        return True
    else:
        print(f"  ⚠ No cells updated (might already be updated)")
        return False

def main():
    """Main execution function"""
    print("=" * 70)
    print("NASA Defect Notebooks - Batch Hyperparameter Update")
    print("=" * 70)
    print("\nUpdating 4 cells per notebook:")
    print("  • cell-10: KAN model (grid_size=10, remove sigmoid)")
    print("  • cell-11: KAN_Attention model (grid_size=10, remove sigmoid)")
    print("  • cell-22: KAN training (pos_weight, BCEWithLogitsLoss)")
    print("  • cell-24: KAN_Attention training (pos_weight, BCEWithLogitsLoss)")
    
    base_path = Path("/home/user/nasa-defect-gwo-kan")
    updated_count = 0
    
    for notebook_file in NOTEBOOKS:
        notebook_path = base_path / notebook_file
        if notebook_path.exists():
            if update_notebook(notebook_path):
                updated_count += 1
        else:
            print(f"\n⚠ Not found: {notebook_path}")
    
    print("\n" + "=" * 70)
    print(f"✓ Batch update complete! Updated {updated_count}/{len(NOTEBOOKS)} notebooks")
    print("=" * 70)

if __name__ == "__main__":
    main()
