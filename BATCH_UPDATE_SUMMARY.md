# Batch Hyperparameter Optimization Update

## Overview
Successfully updated 7 NASA defect experiment notebooks (MC1, MC2, MW1, PC1, PC2, PC3, PC4) with the same hyperparameter optimizations applied to CM1.

## Notebooks Updated
1. experiments/MC1_experiment.ipynb
2. experiments/MC2_experiment.ipynb
3. experiments/MW1_experiment.ipynb
4. experiments/PC1_experiment.ipynb
5. experiments/PC2_experiment.ipynb
6. experiments/PC3_experiment.ipynb
7. experiments/PC4_experiment.ipynb

## Changes Applied (4 cells per notebook)

### Cell 10: KAN Model Definition
- Changed `grid_size=3` → `grid_size=10` (better spline approximation capacity)
- Removed `torch.sigmoid(x)` from forward method
- Changed to `return x` (raw logits for BCEWithLogitsLoss)

### Cell 11: KAN_Attention Model Definition
- Changed `grid_size=3` → `grid_size=10` (better spline approximation capacity)
- Removed `torch.sigmoid(x)` from forward method
- Changed to `return x` (raw logits for BCEWithLogitsLoss)

### Cell 22: KAN Training
- Updated model initialization: `grid_size=10`
- Replaced `FocalLoss(alpha=0.25, gamma=2.0)` with:
  ```python
  pos_weight = torch.tensor([3.5]).to(device)
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  ```
- Added `torch.sigmoid()` to all prediction/inference lines:
  - `torch.sigmoid(model_kan(X_val_tensor))`
  - `torch.sigmoid(model_kan(X_test_tensor))`

### Cell 24: KAN_Attention Training
- Updated model initialization: `grid_size=10`
- Replaced `FocalLoss(alpha=0.25, gamma=2.0)` with:
  ```python
  pos_weight = torch.tensor([3.5]).to(device)
  criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  ```
- Added `torch.sigmoid()` to all prediction/inference lines:
  - `torch.sigmoid(model_kan_att(X_val_tensor))`
  - `torch.sigmoid(model_kan_att(X_test_tensor))`

## Rationale

### Why BCEWithLogitsLoss + pos_weight instead of FocalLoss?
1. **Numerical Stability**: BCEWithLogitsLoss combines sigmoid + BCE in a numerically stable way
2. **Direct Class Balancing**: pos_weight=3.5 makes missing defects (FN) 3.5x costlier than false alarms (FP)
3. **Simpler & More Interpretable**: Direct weight control vs. alpha/gamma hyperparameters
4. **Better Gradient Flow**: No intermediate sigmoid activation

### Why grid_size=10 instead of 3?
- Higher grid_size = more spline basis functions = better approximation capacity
- Allows the KAN layers to learn more complex non-linear transformations
- Minimal computational overhead for the benefit gained

### Why remove sigmoid from model forward() and add to inference?
- BCEWithLogitsLoss expects raw logits (pre-sigmoid)
- During training: logits go directly to loss function
- During inference: manually apply sigmoid to get probabilities
- This separation is the recommended PyTorch pattern for numerical stability

## Automation Scripts Created

1. **batch_update_notebooks.py**: Initial script for model definitions
2. **batch_update_notebooks_v2.py**: Enhanced script handling FocalLoss → BCEWithLogitsLoss
3. **add_sigmoid_to_predictions.py**: Specialized script for adding sigmoid to inference

## Verification
All notebooks verified with comprehensive checks:
- ✓ Model definitions use grid_size=10 and return raw logits
- ✓ Training cells use BCEWithLogitsLoss with pos_weight=3.5
- ✓ Inference/prediction lines properly apply sigmoid activation
- ✓ All 28 cells updated successfully (4 cells × 7 notebooks)

## Next Steps
These notebooks can now be executed to evaluate the improved hyperparameters on all NASA MDP datasets.
