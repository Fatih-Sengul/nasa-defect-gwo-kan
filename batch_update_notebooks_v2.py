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
    """Update training cell: Replace FocalLoss with BCEWithLogitsLoss + pos_weight"""
    
    # Check if already updated
    if 'BCEWithLogitsLoss' in source_code and 'pos_weight' in source_code:
        print("    (Already has BCEWithLogitsLoss with pos_weight)")
        return source_code
    
    # Update grid_size in model initialization
    source_code = re.sub(r'grid_size\s*=\s*3', 'grid_size=10', source_code)
    
    # Replace FocalLoss with pos_weight + BCEWithLogitsLoss
    # Pattern: criterion = FocalLoss(alpha=0.25, gamma=2.0)
    focal_loss_pattern = r'criterion\s*=\s*FocalLoss\([^)]+\)'
    replacement = '''pos_weight = torch.tensor([3.5]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)'''
    
    source_code = re.sub(focal_loss_pattern, replacement, source_code)
    
    # Add sigmoid to predictions
    # Pattern 1: y_pred = model(...) in test section
    source_code = re.sub(
        r'(\s+)(y_pred|predictions)\s*=\s*model\((X_test_tensor|X_test)\)',
        r'\1\2 = torch.sigmoid(model(\3))',
        source_code
    )
    
    # Pattern 2: validation outputs
    source_code = re.sub(
        r'(\s+)(val_outputs|outputs)\s*=\s*model\(X_val_batch\)',
        r'\1\2 = torch.sigmoid(model(X_val_batch))',
        source_code
    )
    
    # Handle .squeeze() cases
    source_code = re.sub(
        r'torch\.sigmoid\(model\(([^)]+)\)\)\.squeeze\(\)',
        r'torch.sigmoid(model(\1)).squeeze()',
        source_code
    )
    
    # Remove double sigmoid if any
    source_code = re.sub(
        r'torch\.sigmoid\(torch\.sigmoid\(model\(([^)]+)\)\)\)',
        r'torch.sigmoid(model(\1))',
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
            if 'class KAN' in original_source:
                updated_source = update_kan_model(original_source)
                if updated_source != original_source:
                    cell_10['source'] = updated_source.splitlines(keepends=True)
                    updated_cells += 1
                    print(f"  ✓ Updated cell-10 (KAN model)")
    
    # Update cell-11 (KAN_Attention model)
    if len(cells) > 11:
        cell_11 = cells[11]
        if cell_11.get('cell_type') == 'code':
            original_source = ''.join(cell_11['source'])
            if 'class KAN_Attention' in original_source:
                updated_source = update_kan_model(original_source)
                if updated_source != original_source:
                    cell_11['source'] = updated_source.splitlines(keepends=True)
                    updated_cells += 1
                    print(f"  ✓ Updated cell-11 (KAN_Attention model)")
    
    # Update cell-22 (KAN training)
    if len(cells) > 22:
        cell_22 = cells[22]
        if cell_22.get('cell_type') == 'code':
            original_source = ''.join(cell_22['source'])
            if 'criterion' in original_source:
                print(f"  • Updating cell-22 (KAN training)...")
                updated_source = update_training_cell(original_source)
                if updated_source != original_source:
                    cell_22['source'] = updated_source.splitlines(keepends=True)
                    updated_cells += 1
                    print(f"  ✓ Updated cell-22 (KAN training)")
    
    # Update cell-24 (KAN_Attention training)
    if len(cells) > 24:
        cell_24 = cells[24]
        if cell_24.get('cell_type') == 'code':
            original_source = ''.join(cell_24['source'])
            if 'criterion' in original_source or 'model_kan_att' in original_source:
                print(f"  • Updating cell-24 (KAN_Attention training)...")
                updated_source = update_training_cell(original_source)
                if updated_source != original_source:
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
    print("NASA Defect Notebooks - Batch Hyperparameter Update V2")
    print("=" * 70)
    print("\nUpdating 4 cells per notebook:")
    print("  • cell-10: KAN model (grid_size=10, remove sigmoid)")
    print("  • cell-11: KAN_Attention model (grid_size=10, remove sigmoid)")
    print("  • cell-22: KAN training (grid_size=10, FocalLoss->BCEWithLogitsLoss)")
    print("  • cell-24: KAN_Attention training (grid_size=10, FocalLoss->BCEWithLogitsLoss)")
    
    base_path = Path("/home/user/nasa-defect-gwo-kan")
    updated_count = 0
    total_cells_updated = 0
    
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
