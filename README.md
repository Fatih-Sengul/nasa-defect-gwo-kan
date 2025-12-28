# Safety-Aware Software Defect Prediction Framework

## GWO-Optimized KAN with SMOTE for NASA MDP Datasets

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

This repository implements a novel **Safety-Aware Software Defect Prediction Framework** designed to outperform existing literature in defect prediction on NASA MDP (Metrics Data Program) datasets. The framework leverages:

- **Custom Kolmogorov-Arnold Networks (KAN)**: A PyTorch-based implementation with learnable spline-based activation functions
- **Grey Wolf Optimizer (GWO)**: Bio-inspired optimization algorithm for hyperparameter tuning
- **SMOTE**: Synthetic Minority Over-sampling Technique to handle class imbalance while preserving data integrity
- **Safety-First Approach**: Optimizes for **Recall** and **F2-Score** rather than accuracy, critical for safety-critical applications

### Key Innovation

Unlike traditional approaches that use **undersampling** (which loses valuable information), this framework uses **SMOTE** to preserve all training data while balancing classes. The GWO dynamically tunes KAN hyperparameters to maximize recall, ensuring defects are detected with minimal false negatives.

---

## Architecture

### 1. Custom KAN Implementation

```
KAN Architecture:
Input (n_features)
   ↓
KANLinear (learnable splines)
   ↓
BatchNorm + ReLU + Dropout
   ↓
KANLinear (learnable splines)
   ↓
BatchNorm + ReLU + Dropout
   ↓
Linear + Sigmoid
   ↓
Output (binary classification)
```

**KANLinear Layer Features**:
- Learnable B-spline basis functions on edges (not nodes)
- Grid-based interpolation with configurable grid size
- Residual connections for training stability
- Polynomial terms for high-order approximations

### 2. Grey Wolf Optimizer

**Optimization Process**:
```
Initialize wolf pack (candidate solutions)
For each iteration:
  1. Evaluate fitness (Recall) for all wolves
  2. Identify Alpha (best), Beta (2nd), Delta (3rd)
  3. Update positions guided by top 3 wolves
  4. Track convergence
Return Alpha position (optimal hyperparameters)
```

**Hyperparameters Optimized**:
- `grid_size`: 3-10 (spline grid resolution)
- `spline_order`: 2-5 (spline polynomial order)
- `hidden_dim`: 16-128 (neural network capacity)
- `learning_rate`: 0.001-0.1 (optimization step size)

### 3. SMOTE Application

**Critical Design Decision**:
```python
# ✅ CORRECT: SMOTE on training set only
X_train, X_test = split(X, y)
X_train_smote, y_train_smote = SMOTE(X_train, y_train)
model.train(X_train_smote, y_train_smote)
model.evaluate(X_test, y_test)  # Original test distribution

# ❌ WRONG: SMOTE before split (causes data leakage)
X_smote, y_smote = SMOTE(X, y)
X_train, X_test = split(X_smote, y_smote)
```

---

## Installation

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Dependencies

```bash
pip install numpy pandas scipy scikit-learn
pip install torch torchvision
pip install imbalanced-learn
pip install matplotlib seaborn
pip install openpyxl
```

Or install all at once:

```bash
pip install numpy pandas scipy scikit-learn torch imbalanced-learn matplotlib seaborn openpyxl
```

---

## Usage

### Quick Start

1. **Ensure datasets are in `./dataset/` directory**:
   ```
   dataset/
   ├── CM1.arff
   ├── JM1.arff
   ├── KC1.arff
   ├── KC3.arff
   ├── KC4.arff
   ├── MC1.arff
   ├── MC2.arff
   ├── MW1.arff
   ├── PC1.arff
   ├── PC2.arff
   ├── PC3.arff
   ├── PC4.arff
   └── PC5.arff
   ```

2. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook main_gwo_kan.ipynb
   ```

3. **Run all cells sequentially**:
   - Cell 1: Import dependencies
   - Cell 2: Define custom KAN architecture
   - Cell 3: Implement GWO optimizer
   - Cell 4: Data loading and preprocessing
   - Cell 5: Model training and evaluation
   - Cell 6: GWO-KAN optimization pipeline
   - Cell 7: Main execution pipeline
   - Cell 8: Execute framework on all datasets
   - Cell 9: Visualization (optional)

4. **Results will be saved to**:
   - `final_results.xlsx` - Detailed metrics for all datasets
   - `gwo_kan_results.png` - Visualization of performance metrics

---

## Execution Pipeline

### For Each Dataset:

```
STEP 1: Load Data
  ├─ Parse .arff file
  ├─ Handle byte-string encoding issues
  └─ Create DataFrame

STEP 2: Preprocessing
  ├─ Separate features and labels
  ├─ Encode categorical labels
  ├─ Handle missing values (median imputation)
  └─ Check class distribution

STEP 3: Train/Test Split + SMOTE
  ├─ Stratified split (80% train, 20% test)
  ├─ MinMax normalization
  └─ Apply SMOTE to training set ONLY

STEP 4: GWO Hyperparameter Optimization
  ├─ Initialize 10 wolves (candidate solutions)
  ├─ Run 20 iterations
  ├─ Evaluate fitness (Recall on validation set)
  └─ Track Alpha, Beta, Delta wolves

STEP 5: Train Final Model
  ├─ Create KAN with optimal hyperparameters
  ├─ Train on full SMOTE-augmented training set
  ├─ Early stopping based on recall
  └─ Save best model

STEP 6: Evaluation on Test Set
  ├─ Calculate Accuracy, Precision, Recall
  ├─ Calculate F1-Score, F2-Score, AUC
  └─ Store results
```

---

## Results Format

### Excel Output (`final_results.xlsx`)

| Dataset | Samples | Features | Grid_Size | Spline_Order | Hidden_Dim | Learning_Rate | Accuracy | Precision | Recall | F1-Score | F2-Score | AUC |
|---------|---------|----------|-----------|--------------|------------|---------------|----------|-----------|--------|----------|----------|-----|
| CM1     | 505     | 37       | 5         | 3            | 64         | 0.0123        | 0.8712   | 0.8234    | 0.9156 | 0.8672   | 0.8945   | 0.91|
| JM1     | 10885   | 21       | 7         | 4            | 96         | 0.0089        | 0.8523   | 0.8012    | 0.9234 | 0.8589   | 0.8912   | 0.88|
| ...     | ...     | ...      | ...       | ...          | ...        | ...           | ...      | ...       | ...    | ...      | ...      | ... |
| AVERAGE | -       | -        | -         | -            | -          | -             | 0.8634   | 0.8145    | 0.9187 | 0.8645   | 0.8934   | 0.89|

### Metrics Explanation

- **Accuracy**: Overall correctness (not prioritized in safety-critical contexts)
- **Precision**: Of predicted defects, how many are truly defective
- **Recall** ⭐: Of all actual defects, how many did we detect (SAFETY METRIC)
- **F1-Score**: Harmonic mean of precision and recall
- **F2-Score**: Weighted F-score favoring recall (β=2)
- **AUC**: Area under ROC curve (model discrimination ability)

---

## Technical Details

### Why KAN over MLP?

**Kolmogorov-Arnold Networks** represent functions as sums of compositions of univariate functions, aligning with the Kolmogorov-Arnold representation theorem:

```
f(x₁, ..., xₙ) = Σᵢ Φᵢ(Σⱼ φᵢⱼ(xⱼ))
```

**Advantages**:
- Learnable activation functions (splines) instead of fixed ReLU/Sigmoid
- Better interpretability for understanding feature interactions
- Potentially better generalization on structured/tabular data

### Why GWO for Optimization?

**Grey Wolf Optimizer** mimics the hunting behavior and social hierarchy of grey wolves:

1. **Hierarchy**: Alpha (best), Beta (2nd), Delta (3rd), Omega (rest)
2. **Encircling**: Wolves surround prey (optimal solution)
3. **Hunting**: Guided by top 3 wolves (exploitation)
4. **Exploration**: Random components prevent local minima

**Benefits**:
- Fewer hyperparameters than GA/PSO
- Good balance between exploration and exploitation
- Naturally handles continuous and discrete parameters

### Why SMOTE over Undersampling?

**Undersampling** (used in Turk & Coşkuncay, 2025):
- ❌ Discards majority class samples → **information loss**
- ❌ Reduced training set size
- ❌ May remove important boundary cases

**SMOTE** (our approach):
- ✅ Generates synthetic minority samples → **preserves all original data**
- ✅ Larger training set
- ✅ Better decision boundaries

---

## Comparison to Existing Literature

### Turk & Coşkuncay (2025)
- **Method**: Undersampling + Fixed KAN hyperparameters
- **Issue**: Data loss from undersampling, non-optimized KAN
- **Metrics**: Focused on accuracy

### Our Framework
- **Method**: SMOTE + GWO-optimized KAN
- **Advantage**: Full data preservation, adaptive hyperparameters
- **Metrics**: **Recall-first** approach for safety

**Expected Improvements**:
- Higher Recall (fewer missed defects)
- Better F2-Score (safety emphasis)
- Competitive or better AUC

---

## File Structure

```
nasa-defect-gwo-kan/
├── dataset/                  # NASA MDP .arff files
│   ├── CM1.arff
│   ├── JM1.arff
│   └── ...
├── main_gwo_kan.ipynb       # Main Jupyter notebook
├── final_results.xlsx       # Generated results
├── gwo_kan_results.png      # Generated visualization
├── README.md                # This file
└── requirements.txt         # Python dependencies
```

---

## Reproducibility

**Random Seed**: 42 (set for NumPy, PyTorch, scikit-learn)

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
```

**Hardware**: CPU or CUDA-enabled GPU (auto-detected)

**Expected Runtime**:
- Per dataset: ~5-15 minutes (depending on size and GWO iterations)
- All 13 datasets: ~1-3 hours (with 20 GWO iterations)

**Optimization Tips**:
- Reduce `n_iterations` in GWO (20 → 10) for faster experimentation
- Reduce `epochs` in training (100 → 50) for quicker convergence
- Use GPU if available for significant speedup

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{gwo_kan_framework_2025,
  title={Safety-Aware Software Defect Prediction Framework: GWO-Optimized KAN with SMOTE},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/nasa-defect-gwo-kan}
}
```

---

## Future Enhancements

- [ ] Multi-objective optimization (Recall + Precision simultaneously)
- [ ] Ensemble of GWO-optimized KANs
- [ ] Explainability module (SHAP/LIME for KAN)
- [ ] Transfer learning across NASA datasets
- [ ] Integration with CI/CD pipelines for real-time defect prediction

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **NASA MDP**: For providing high-quality software metrics datasets
- **Turk & Coşkuncay (2025)**: Baseline KAN approach for defect prediction
- **Mirjalili et al. (2014)**: Original Grey Wolf Optimizer paper
- **PyTorch Team**: Deep learning framework
- **imbalanced-learn**: SMOTE implementation

---

## Contact

For questions, issues, or collaboration:
- Open an issue on GitHub
- Email: your.email@example.com

---

**Built with ❤️ for safer software systems**
