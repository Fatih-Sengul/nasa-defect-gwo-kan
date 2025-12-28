# Safety-Aware Software Defect Prediction Framework

## 🎯 Objective

Develop a novel defect prediction framework that **maximizes Recall** (safety metric) for NASA MDP datasets using:

- **SMOTE** (Synthetic Minority Over-sampling Technique) - preserves data instead of undersampling
- **GWO** (Grey Wolf Optimizer) - metaheuristic optimization for hyperparameter tuning
- **KAN** (Kolmogorov-Arnold Networks) - neural architecture for non-linear pattern learning

## 🔬 Research Contribution

This framework aims to outperform existing approaches (e.g., Turk & Coşkuncay, 2025) by:

1. **Preserving Information**: Using SMOTE instead of undersampling
2. **Adaptive Optimization**: GWO tunes KAN hyperparameters for each dataset
3. **Safety Focus**: Optimizing Recall as the primary metric (critical for defect detection)

## 📊 Architecture

```
NASA MDP Dataset (.arff)
    ↓
Data Loading & Preprocessing
    ↓
Train/Test Split (80/20 Stratified)
    ↓
SMOTE on Training Set ONLY
    ↓
Grey Wolf Optimizer
    ├─ Optimizes: grid_size, spline_order, hidden_dim, learning_rate
    └─ Objective: Maximize Recall
    ↓
Train Final KAN Model
    ↓
Evaluate on Test Set
    ↓
Save Results (final_results.xlsx)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Datasets

Place NASA MDP `.arff` files in the `./dataset/` directory:

```
dataset/
├── CM1.arff
├── JM1.arff
├── KC1.arff
├── KC2.arff
├── MC2.arff
└── PC1.arff
```

### 3. Run the Pipeline

Open and run the Jupyter notebook:

```bash
jupyter notebook main_gwo_kan.ipynb
```

Or execute all cells programmatically:

```bash
jupyter nbconvert --to notebook --execute main_gwo_kan.ipynb
```

## 📁 Project Structure

```
nasa-defect-gwo-kan/
├── main_gwo_kan.ipynb      # Complete pipeline notebook
├── dataset/                # NASA MDP datasets (.arff files)
├── final_results.xlsx      # Output results (generated after run)
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🧩 Key Components

### 1. **KAN (Kolmogorov-Arnold Network)**

- Custom PyTorch implementation
- Uses B-spline basis functions (approximated with RBF kernels)
- 3-layer architecture with dropout regularization
- Learns complex non-linear transformations

### 2. **Grey Wolf Optimizer (GWO)**

- Population-based metaheuristic algorithm
- Simulates wolf pack hunting behavior (Alpha, Beta, Delta hierarchy)
- Optimizes 4 hyperparameters:
  - `grid_size`: [3, 10]
  - `spline_order`: [2, 5]
  - `hidden_dim`: [16, 128]
  - `learning_rate`: [0.001, 0.1]

### 3. **Data Pipeline**

- **ARFF Loading**: Handles NASA MDP format with byte-string decoding
- **Preprocessing**: Feature normalization, label encoding, missing value imputation
- **SMOTE**: Balances training data without information loss
- **Stratified Split**: Maintains class distribution in train/test sets

## 📈 Output Metrics

For each dataset, the framework reports:

- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate (PRIMARY SAFETY METRIC)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

Results are saved to `final_results.xlsx` with:

- Best hyperparameters for each dataset
- Test set performance metrics
- Summary statistics (average Recall, F1-Score, etc.)

## 🔍 Usage Example

```python
# Initialize the predictor
predictor = SafetyAwareDefectPredictor(dataset_dir='./dataset/')

# Process all datasets
predictor.run_all_datasets()

# Results are automatically saved to final_results.xlsx
```

## 🛡️ Safety-First Design

This framework prioritizes **Recall** because:

- In safety-critical software, **false negatives are costly** (missing defects)
- High recall ensures most defective modules are detected
- GWO explicitly optimizes for Recall during hyperparameter tuning

## 📚 Requirements

- Python 3.10+
- PyTorch
- Scikit-learn
- Imbalanced-learn
- Pandas
- NumPy
- SciPy
- openpyxl (for Excel export)

See `requirements.txt` for exact versions.

## 🎓 Citation

If you use this framework in your research, please cite:

```
Safety-Aware Software Defect Prediction using GWO-optimized KAN with SMOTE
GitHub: https://github.com/Fatih-Sengul/nasa-defect-gwo-kan
```

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Project Issues](https://github.com/Fatih-Sengul/nasa-defect-gwo-kan/issues)

---

**Note**: This is a research framework. Performance may vary across different NASA MDP datasets. Always validate results on your specific use case.
