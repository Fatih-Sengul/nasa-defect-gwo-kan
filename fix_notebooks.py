#!/usr/bin/env python3
"""
Script to restructure NASA defect experiment notebooks into modular, executable cells.
Splits large code blocks into smaller, logical steps.
"""

import json
import os
from pathlib import Path


def create_modular_notebook(dataset_name):
    """Create a properly structured notebook for the given dataset."""

    cells = []

    # Cell 0: Title (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# 🚀 NASA Defect Prediction: {dataset_name}\n",
            "\n",
            f"**Dataset:** {dataset_name}\n",
            "**Method:** Baseline RF → KAN Base → KAN + Attention\n",
            "**Goal:** F2 & Recall optimization (defect detection)\n",
            "\n",
            "**✅ Self-contained:** Run all cells in order, no dependencies!\n",
            "\n",
            "---"
        ]
    })

    # Cell 1: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📦 Step 1: Setup & Environment"]
    })

    # Cell 2: Google Drive Mount (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Mount Google Drive\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    print('✅ Google Drive mounted!')\n",
            "except ImportError:\n",
            "    print('⚠️  Not on Colab - skipping mount')\n"
        ]
    })

    # Cell 3: Package Installation (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Install required packages\n",
            "import sys\n",
            "!{sys.executable} -m pip install imbalanced-learn scipy scikit-learn torch matplotlib seaborn pandas numpy openpyxl -q\n",
            "print('✅ Packages installed!')\n"
        ]
    })

    # Cell 4: Imports (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Import libraries\n",
            "import os\n",
            "import json\n",
            "import warnings\n",
            "import datetime\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from scipy.io import arff\n",
            "from io import StringIO\n",
            "\n",
            "from sklearn.preprocessing import MinMaxScaler, LabelEncoder\n",
            "from sklearn.model_selection import train_test_split\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.metrics import (\n",
            "    accuracy_score, precision_score, recall_score, f1_score,\n",
            "    fbeta_score, confusion_matrix, average_precision_score\n",
            ")\n",
            "from imblearn.over_sampling import SMOTE\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "import torch.optim as optim\n",
            "from torch.utils.data import TensorDataset, DataLoader\n",
            "\n",
            "warnings.filterwarnings('ignore')\n",
            "print('✅ Imports complete!')\n"
        ]
    })

    # Cell 5: Configuration (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Configuration\n",
            f"DATASET_NAME = '{dataset_name}'\n",
            "DATASET_PATH = '/content/drive/MyDrive/nasa-defect-gwo-kan/dataset'\n",
            "OUTPUT_DIR = f'./results_{DATASET_NAME}'\n",
            "SEED = 42\n",
            "\n",
            "# Set seeds for reproducibility\n",
            "np.random.seed(SEED)\n",
            "torch.manual_seed(SEED)\n",
            "device = torch.device('cpu')\n",
            "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
            "\n",
            "RUN_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')\n",
            "\n",
            "print(f'✅ Configuration complete!')\n",
            f"print(f'📊 Dataset: {dataset_name}')\n",
            "print(f'🖥️  Device: {device}')\n",
            "print(f'📁 Output: {OUTPUT_DIR}')\n"
        ]
    })

    # Cell 6: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🛠️ Step 2: Define Utility Functions"]
    })

    # Cell 7: Utility Functions (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Utility functions for data loading and metrics\n",
            "\n",
            "def load_arff(file_path):\n",
            "    \"\"\"Load ARFF file and return pandas DataFrame.\"\"\"\n",
            "    try:\n",
            "        data, _ = arff.loadarff(file_path)\n",
            "        df = pd.DataFrame(data)\n",
            "        for col in df.columns:\n",
            "            if df[col].dtype == object:\n",
            "                try:\n",
            "                    df[col] = df[col].str.decode('utf-8')\n",
            "                except:\n",
            "                    pass\n",
            "        return df\n",
            "    except:\n",
            "        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:\n",
            "            content = f.read()\n",
            "        data_start = content.lower().find('@data')\n",
            "        data_section = content[data_start + 5:].strip()\n",
            "        return pd.read_csv(StringIO(data_section), header=None)\n",
            "\n",
            "def calc_metrics(y_true, y_pred, y_proba=None):\n",
            "    \"\"\"Calculate comprehensive metrics for defect prediction.\"\"\"\n",
            "    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()\n",
            "    m = {\n",
            "        'recall': recall_score(y_true, y_pred, zero_division=0),\n",
            "        'precision': precision_score(y_true, y_pred, zero_division=0),\n",
            "        'f1': f1_score(y_true, y_pred, zero_division=0),\n",
            "        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),\n",
            "        'accuracy': accuracy_score(y_true, y_pred),\n",
            "        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)\n",
            "    }\n",
            "    if y_proba is not None:\n",
            "        try:\n",
            "            m['pr_auc'] = average_precision_score(y_true, y_proba)\n",
            "        except:\n",
            "            m['pr_auc'] = 0.0\n",
            "    else:\n",
            "        m['pr_auc'] = 0.0\n",
            "    return m\n",
            "\n",
            "def find_threshold(y_true, y_proba):\n",
            "    \"\"\"Find optimal threshold for F2 score.\"\"\"\n",
            "    best_score, best_t = -1, 0.5\n",
            "    for t in np.arange(0.05, 0.96, 0.05):\n",
            "        y_pred = (y_proba >= t).astype(int)\n",
            "        m = calc_metrics(y_true, y_pred)\n",
            "        score = m['f2'] if m['accuracy'] >= 0.5 else 0\n",
            "        if score > best_score:\n",
            "            best_score, best_t = score, t\n",
            "    return best_t\n",
            "\n",
            "print('✅ Utility functions defined!')\n"
        ]
    })

    # Cell 8: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🧠 Step 3: Define KAN Model Architecture"]
    })

    # Cell 9: KANLinear (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# KANLinear: Core KAN layer with spline-based activation\n",
            "\n",
            "class KANLinear(nn.Module):\n",
            "    \"\"\"Kolmogorov-Arnold Network Linear Layer with learnable spline activations.\"\"\"\n",
            "    \n",
            "    def __init__(self, in_features, out_features, grid_size=3, spline_order=2):\n",
            "        super().__init__()\n",
            "        self.in_features = in_features\n",
            "        self.out_features = out_features\n",
            "        self.grid_size = grid_size\n",
            "        self.spline_order = spline_order\n",
            "        \n",
            "        # Learnable grid points\n",
            "        self.grid = nn.Parameter(\n",
            "            torch.linspace(-1, 1, grid_size)\n",
            "            .unsqueeze(0).unsqueeze(0)\n",
            "            .repeat(out_features, in_features, 1)\n",
            "        )\n",
            "        \n",
            "        # Spline coefficients\n",
            "        self.coef = nn.Parameter(\n",
            "            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1\n",
            "        )\n",
            "        \n",
            "        # Base weights\n",
            "        self.base_weight = nn.Parameter(\n",
            "            torch.randn(out_features, in_features) * 0.1\n",
            "        )\n",
            "    \n",
            "    def forward(self, x):\n",
            "        batch_size = x.shape[0]\n",
            "        \n",
            "        # Expand input for broadcasting\n",
            "        x_expanded = x.unsqueeze(1).unsqueeze(-1)  # [B, 1, in_f, 1]\n",
            "        \n",
            "        # Compute distances to grid points\n",
            "        distances = torch.abs(x_expanded - self.grid.unsqueeze(0))\n",
            "        \n",
            "        # Build basis functions\n",
            "        basis = torch.zeros(\n",
            "            batch_size, self.out_features, self.in_features,\n",
            "            self.grid_size + self.spline_order,\n",
            "            device=x.device\n",
            "        )\n",
            "        \n",
            "        # RBF basis for grid points\n",
            "        for i in range(self.grid_size):\n",
            "            basis[:, :, :, i] = torch.exp(-distances[:, :, :, i] ** 2 / 0.5)\n",
            "        \n",
            "        # Polynomial basis\n",
            "        for i in range(self.spline_order):\n",
            "            basis[:, :, :, self.grid_size + i] = x_expanded.squeeze(-1) ** (i + 1)\n",
            "        \n",
            "        # Compute spline output\n",
            "        spline_output = (basis * self.coef.unsqueeze(0)).sum(dim=-1).sum(dim=-1)\n",
            "        \n",
            "        # Add base transformation\n",
            "        base_output = torch.matmul(x, self.base_weight.t())\n",
            "        \n",
            "        return spline_output + base_output\n",
            "\n",
            "print('✅ KANLinear layer defined!')\n"
        ]
    })

    # Cell 10: KAN Model (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# KAN: Complete KAN model for defect prediction\n",
            "\n",
            "class KAN(nn.Module):\n",
            "    \"\"\"Kolmogorov-Arnold Network for binary defect classification.\"\"\"\n",
            "    \n",
            "    def __init__(self, input_dim, hidden_dim=32, grid_size=3, spline_order=2):\n",
            "        super().__init__()\n",
            "        self.kan1 = KANLinear(input_dim, hidden_dim, grid_size, spline_order)\n",
            "        self.kan2 = KANLinear(hidden_dim, hidden_dim // 2, grid_size, spline_order)\n",
            "        self.output_layer = nn.Linear(hidden_dim // 2, 1)\n",
            "        \n",
            "        self.bn1 = nn.BatchNorm1d(hidden_dim)\n",
            "        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)\n",
            "        self.dropout = nn.Dropout(0.3)\n",
            "    \n",
            "    def forward(self, x):\n",
            "        x = self.kan1(x)\n",
            "        x = self.bn1(x)\n",
            "        x = F.relu(x)\n",
            "        x = self.dropout(x)\n",
            "        \n",
            "        x = self.kan2(x)\n",
            "        x = self.bn2(x)\n",
            "        x = F.relu(x)\n",
            "        x = self.dropout(x)\n",
            "        \n",
            "        x = self.output_layer(x)\n",
            "        return torch.sigmoid(x)\n",
            "\n",
            "print('✅ KAN model defined!')\n"
        ]
    })

    # Cell 11: Attention & KAN_Attention (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Attention mechanism and KAN with Attention\n",
            "\n",
            "class Attention(nn.Module):\n",
            "    \"\"\"Feature attention mechanism.\"\"\"\n",
            "    \n",
            "    def __init__(self, input_dim, attention_dim=16):\n",
            "        super().__init__()\n",
            "        self.fc1 = nn.Linear(input_dim, attention_dim)\n",
            "        self.fc2 = nn.Linear(attention_dim, input_dim)\n",
            "        self.dropout = nn.Dropout(0.2)\n",
            "    \n",
            "    def forward(self, x):\n",
            "        attention_weights = F.relu(self.fc1(x))\n",
            "        attention_weights = self.dropout(attention_weights)\n",
            "        attention_weights = torch.sigmoid(self.fc2(attention_weights))\n",
            "        return x * attention_weights, attention_weights\n",
            "\n",
            "\n",
            "class KAN_Attention(nn.Module):\n",
            "    \"\"\"KAN with attention mechanism for enhanced feature selection.\"\"\"\n",
            "    \n",
            "    def __init__(self, input_dim, hidden_dim=32, grid_size=3, spline_order=2):\n",
            "        super().__init__()\n",
            "        self.attention = Attention(input_dim, attention_dim=16)\n",
            "        self.kan1 = KANLinear(input_dim, hidden_dim, grid_size, spline_order)\n",
            "        self.kan2 = KANLinear(hidden_dim, hidden_dim // 2, grid_size, spline_order)\n",
            "        self.output_layer = nn.Linear(hidden_dim // 2, 1)\n",
            "        \n",
            "        self.bn1 = nn.BatchNorm1d(hidden_dim)\n",
            "        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)\n",
            "        self.dropout = nn.Dropout(0.3)\n",
            "    \n",
            "    def forward(self, x):\n",
            "        # Apply attention\n",
            "        x_attended, _ = self.attention(x)\n",
            "        \n",
            "        # KAN layers\n",
            "        x = self.kan1(x_attended)\n",
            "        x = self.bn1(x)\n",
            "        x = F.relu(x)\n",
            "        x = self.dropout(x)\n",
            "        \n",
            "        x = self.kan2(x)\n",
            "        x = self.bn2(x)\n",
            "        x = F.relu(x)\n",
            "        x = self.dropout(x)\n",
            "        \n",
            "        x = self.output_layer(x)\n",
            "        return torch.sigmoid(x)\n",
            "\n",
            "print('✅ Attention models defined!')\n"
        ]
    })

    # Cell 12: Focal Loss (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Focal Loss for imbalanced classification\n",
            "\n",
            "class FocalLoss(nn.Module):\n",
            "    \"\"\"Focal Loss to handle class imbalance.\"\"\"\n",
            "    \n",
            "    def __init__(self, alpha=0.25, gamma=2.0):\n",
            "        super().__init__()\n",
            "        self.alpha = alpha\n",
            "        self.gamma = gamma\n",
            "    \n",
            "    def forward(self, inputs, targets):\n",
            "        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')\n",
            "        pt = torch.exp(-bce_loss)\n",
            "        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss\n",
            "        return focal_loss.mean()\n",
            "\n",
            "print('✅ Focal Loss defined!')\n"
        ]
    })

    # Cell 13: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📊 Step 4: Load and Preprocess Data"]
    })

    # Cell 14: Data Loading (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Load dataset\n",
            f"print('\\n' + '='*70)\n",
            f"print('🚀 LOADING DATASET: {dataset_name}')\n",
            "print('='*70 + '\\n')\n",
            "\n",
            f"file_path = os.path.join(DATASET_PATH, '{dataset_name}.arff')\n",
            "df = load_arff(file_path)\n",
            "\n",
            "# Separate features and labels\n",
            "X = df.iloc[:, :-1].values.astype(np.float32)\n",
            "y = df.iloc[:, -1].values\n",
            "\n",
            "# Encode labels if needed\n",
            "if y.dtype == object:\n",
            "    y = LabelEncoder().fit_transform(y)\n",
            "else:\n",
            "    y = y.astype(int)\n",
            "\n",
            "print(f'✅ Dataset loaded successfully!')\n",
            "print(f'   Total samples: {len(y)}')\n",
            "print(f'   Features: {X.shape[1]}')\n",
            "print(f'   Defective samples: {np.sum(y==1)} ({np.mean(y==1):.2%})')\n",
            "print(f'   Non-defective samples: {np.sum(y==0)} ({np.mean(y==0):.2%})')\n"
        ]
    })

    # Cell 15: Handle missing values (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Handle missing values (NaN imputation)\n",
            "if np.any(np.isnan(X)):\n",
            "    print('⚠️  Found NaN values, imputing with column medians...')\n",
            "    col_medians = np.nanmedian(X, axis=0)\n",
            "    nan_indices = np.where(np.isnan(X))\n",
            "    X[nan_indices] = np.take(col_medians, nan_indices[1])\n",
            "    print('✅ NaN values imputed!')\n",
            "else:\n",
            "    print('✅ No NaN values found!')\n"
        ]
    })

    # Cell 16: Data Splitting (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Split data (train/val/test) - leakage-free splits\n",
            "X_train_full, X_test, y_train_full, y_test = train_test_split(\n",
            "    X, y, test_size=0.2, stratify=y, random_state=SEED\n",
            ")\n",
            "\n",
            "X_train, X_val, y_train, y_val = train_test_split(\n",
            "    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=SEED\n",
            ")\n",
            "\n",
            "print('✅ Data split complete!')\n",
            "print(f'   Training samples: {len(y_train)}')\n",
            "print(f'   Validation samples: {len(y_val)}')\n",
            "print(f'   Test samples: {len(y_test)}')\n"
        ]
    })

    # Cell 17: Scaling (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Feature scaling (fit on train, transform all sets)\n",
            "scaler = MinMaxScaler()\n",
            "X_train = scaler.fit_transform(X_train)\n",
            "X_val = scaler.transform(X_val)\n",
            "X_test = scaler.transform(X_test)\n",
            "\n",
            "print('✅ Feature scaling complete!')\n"
        ]
    })

    # Cell 18: SMOTE (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Apply SMOTE to training data only (prevent data leakage)\n",
            "smote = SMOTE(sampling_strategy=0.7, random_state=SEED)\n",
            "X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)\n",
            "\n",
            "print('✅ SMOTE resampling complete!')\n",
            "print(f'   Before: {len(y_train)} samples')\n",
            "print(f'   After: {len(y_train_resampled)} samples')\n",
            "print(f'   Added: {len(y_train_resampled) - len(y_train)} synthetic samples')\n",
            "\n",
            "# Initialize results dictionary\n",
            "results = {}\n"
        ]
    })

    # Cell 19: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🌲 Step 5: Train Baseline Random Forest"]
    })

    # Cell 20: Baseline RF Training (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train baseline Random Forest classifier\n",
            "print('\\n' + '='*70)\n",
            "print('🌲 TRAINING BASELINE: RANDOM FOREST')\n",
            "print('='*70 + '\\n')\n",
            "\n",
            "rf_model = RandomForestClassifier(\n",
            "    n_estimators=100,\n",
            "    max_depth=10,\n",
            "    class_weight='balanced',\n",
            "    random_state=SEED,\n",
            "    n_jobs=-1\n",
            ")\n",
            "\n",
            "rf_model.fit(X_train_resampled, y_train_resampled)\n",
            "print('✅ Random Forest training complete!')\n",
            "\n",
            "# Find optimal threshold on validation set\n",
            "y_val_proba_rf = rf_model.predict_proba(X_val)[:, 1]\n",
            "threshold_rf = find_threshold(y_val, y_val_proba_rf)\n",
            "print(f'   Optimal threshold: {threshold_rf:.2f}')\n",
            "\n",
            "# Evaluate on test set\n",
            "y_test_proba_rf = rf_model.predict_proba(X_test)[:, 1]\n",
            "y_test_pred_rf = (y_test_proba_rf >= threshold_rf).astype(int)\n",
            "metrics_rf = calc_metrics(y_test, y_test_pred_rf, y_test_proba_rf)\n",
            "\n",
            "print(f'\\n📊 Test Set Results:')\n",
            "print(f'   Recall:    {metrics_rf[\"recall\"]:.4f}')\n",
            "print(f'   Precision: {metrics_rf[\"precision\"]:.4f}')\n",
            "print(f'   F1 Score:  {metrics_rf[\"f1\"]:.4f}')\n",
            "print(f'   F2 Score:  {metrics_rf[\"f2\"]:.4f}')\n",
            "print(f'   Accuracy:  {metrics_rf[\"accuracy\"]:.4f}')\n",
            "print(f'   PR-AUC:    {metrics_rf[\"pr_auc\"]:.4f}')\n",
            "\n",
            "results['Baseline_RF'] = {'metrics': metrics_rf, 'threshold': threshold_rf}\n"
        ]
    })

    # Cell 21: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🔥 Step 6: Train KAN Base Model"]
    })

    # Cell 22: KAN Base Training (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train KAN base model\n",
            "print('\\n' + '='*70)\n",
            "print('🔥 TRAINING KAN BASE MODEL')\n",
            "print('='*70 + '\\n')\n",
            "\n",
            "# Initialize model\n",
            "model_kan = KAN(\n",
            "    input_dim=X.shape[1],\n",
            "    hidden_dim=32,\n",
            "    grid_size=3,\n",
            "    spline_order=2\n",
            ").to(device)\n",
            "\n",
            "optimizer = optim.Adam(model_kan.parameters(), lr=0.01)\n",
            "criterion = FocalLoss(alpha=0.25, gamma=2.0)\n",
            "\n",
            "# Prepare data loaders\n",
            "X_train_tensor = torch.FloatTensor(X_train_resampled).to(device)\n",
            "y_train_tensor = torch.FloatTensor(y_train_resampled).unsqueeze(1).to(device)\n",
            "X_val_tensor = torch.FloatTensor(X_val).to(device)\n",
            "\n",
            "train_dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
            "train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)\n",
            "\n",
            "# Training loop with early stopping\n",
            "best_val_f2 = 0\n",
            "patience_counter = 0\n",
            "max_patience = 10\n",
            "\n",
            "for epoch in range(50):\n",
            "    model_kan.train()\n",
            "    epoch_loss = 0\n",
            "    \n",
            "    for batch_X, batch_y in train_loader:\n",
            "        optimizer.zero_grad()\n",
            "        outputs = model_kan(batch_X)\n",
            "        loss = criterion(outputs, batch_y)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        epoch_loss += loss.item()\n",
            "    \n",
            "    # Validation\n",
            "    model_kan.eval()\n",
            "    with torch.no_grad():\n",
            "        val_outputs = model_kan(X_val_tensor).cpu().numpy().flatten()\n",
            "        val_predictions = (val_outputs >= 0.5).astype(int)\n",
            "        val_f2 = fbeta_score(y_val, val_predictions, beta=2, zero_division=0)\n",
            "    \n",
            "    if val_f2 > best_val_f2:\n",
            "        best_val_f2 = val_f2\n",
            "        best_model_state = model_kan.state_dict().copy()\n",
            "        patience_counter = 0\n",
            "    else:\n",
            "        patience_counter += 1\n",
            "    \n",
            "    if patience_counter >= max_patience:\n",
            "        print(f'   Early stopping at epoch {epoch + 1}')\n",
            "        model_kan.load_state_dict(best_model_state)\n",
            "        break\n",
            "\n",
            "print(f'✅ KAN training complete!')\n",
            "print(f'   Best validation F2: {best_val_f2:.4f}')\n",
            "\n",
            "# Evaluate on test set\n",
            "model_kan.eval()\n",
            "with torch.no_grad():\n",
            "    y_val_proba_kan = model_kan(X_val_tensor).cpu().numpy().flatten()\n",
            "    X_test_tensor = torch.FloatTensor(X_test).to(device)\n",
            "    y_test_proba_kan = model_kan(X_test_tensor).cpu().numpy().flatten()\n",
            "\n",
            "threshold_kan = find_threshold(y_val, y_val_proba_kan)\n",
            "print(f'   Optimal threshold: {threshold_kan:.2f}')\n",
            "\n",
            "y_test_pred_kan = (y_test_proba_kan >= threshold_kan).astype(int)\n",
            "metrics_kan = calc_metrics(y_test, y_test_pred_kan, y_test_proba_kan)\n",
            "\n",
            "print(f'\\n📊 Test Set Results:')\n",
            "print(f'   Recall:    {metrics_kan[\"recall\"]:.4f}')\n",
            "print(f'   Precision: {metrics_kan[\"precision\"]:.4f}')\n",
            "print(f'   F1 Score:  {metrics_kan[\"f1\"]:.4f}')\n",
            "print(f'   F2 Score:  {metrics_kan[\"f2\"]:.4f}')\n",
            "print(f'   Accuracy:  {metrics_kan[\"accuracy\"]:.4f}')\n",
            "print(f'   PR-AUC:    {metrics_kan[\"pr_auc\"]:.4f}')\n",
            "\n",
            "results['KAN_Base'] = {'metrics': metrics_kan, 'threshold': threshold_kan}\n"
        ]
    })

    # Cell 23: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 🌟 Step 7: Train KAN + Attention Model"]
    })

    # Cell 24: KAN + Attention Training (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train KAN with Attention mechanism\n",
            "print('\\n' + '='*70)\n",
            "print('🌟 TRAINING KAN + ATTENTION MODEL')\n",
            "print('='*70 + '\\n')\n",
            "\n",
            "# Initialize model\n",
            "model_kan_att = KAN_Attention(\n",
            "    input_dim=X.shape[1],\n",
            "    hidden_dim=32,\n",
            "    grid_size=3,\n",
            "    spline_order=2\n",
            ").to(device)\n",
            "\n",
            "optimizer = optim.Adam(model_kan_att.parameters(), lr=0.01)\n",
            "criterion = FocalLoss(alpha=0.25, gamma=2.0)\n",
            "\n",
            "# Training loop with early stopping\n",
            "best_val_f2 = 0\n",
            "patience_counter = 0\n",
            "max_patience = 10\n",
            "\n",
            "for epoch in range(50):\n",
            "    model_kan_att.train()\n",
            "    epoch_loss = 0\n",
            "    \n",
            "    for batch_X, batch_y in train_loader:\n",
            "        optimizer.zero_grad()\n",
            "        outputs = model_kan_att(batch_X)\n",
            "        loss = criterion(outputs, batch_y)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        epoch_loss += loss.item()\n",
            "    \n",
            "    # Validation\n",
            "    model_kan_att.eval()\n",
            "    with torch.no_grad():\n",
            "        val_outputs = model_kan_att(X_val_tensor).cpu().numpy().flatten()\n",
            "        val_predictions = (val_outputs >= 0.5).astype(int)\n",
            "        val_f2 = fbeta_score(y_val, val_predictions, beta=2, zero_division=0)\n",
            "    \n",
            "    if val_f2 > best_val_f2:\n",
            "        best_val_f2 = val_f2\n",
            "        best_model_state = model_kan_att.state_dict().copy()\n",
            "        patience_counter = 0\n",
            "    else:\n",
            "        patience_counter += 1\n",
            "    \n",
            "    if patience_counter >= max_patience:\n",
            "        print(f'   Early stopping at epoch {epoch + 1}')\n",
            "        model_kan_att.load_state_dict(best_model_state)\n",
            "        break\n",
            "\n",
            "print(f'✅ KAN + Attention training complete!')\n",
            "print(f'   Best validation F2: {best_val_f2:.4f}')\n",
            "\n",
            "# Evaluate on test set\n",
            "model_kan_att.eval()\n",
            "with torch.no_grad():\n",
            "    y_val_proba_att = model_kan_att(X_val_tensor).cpu().numpy().flatten()\n",
            "    y_test_proba_att = model_kan_att(X_test_tensor).cpu().numpy().flatten()\n",
            "\n",
            "threshold_att = find_threshold(y_val, y_val_proba_att)\n",
            "print(f'   Optimal threshold: {threshold_att:.2f}')\n",
            "\n",
            "y_test_pred_att = (y_test_proba_att >= threshold_att).astype(int)\n",
            "metrics_att = calc_metrics(y_test, y_test_pred_att, y_test_proba_att)\n",
            "\n",
            "print(f'\\n📊 Test Set Results:')\n",
            "print(f'   Recall:    {metrics_att[\"recall\"]:.4f}')\n",
            "print(f'   Precision: {metrics_att[\"precision\"]:.4f}')\n",
            "print(f'   F1 Score:  {metrics_att[\"f1\"]:.4f}')\n",
            "print(f'   F2 Score:  {metrics_att[\"f2\"]:.4f}')\n",
            "print(f'   Accuracy:  {metrics_att[\"accuracy\"]:.4f}')\n",
            "print(f'   PR-AUC:    {metrics_att[\"pr_auc\"]:.4f}')\n",
            "\n",
            "results['KAN_Attention'] = {'metrics': metrics_att, 'threshold': threshold_att}\n"
        ]
    })

    # Cell 25: Section header (markdown)
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 📈 Step 8: Compare Results & Export"]
    })

    # Cell 26: Final Results & Export (code)
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Compile and display final results\n",
            "print('\\n' + '='*70)\n",
            f"print('📊 FINAL RESULTS - {dataset_name}')\n",
            "print('='*70 + '\\n')\n",
            "\n",
            "results_list = []\n",
            "for model_name, data in results.items():\n",
            "    m = data['metrics']\n",
            "    results_list.append({\n",
            f"        'dataset': '{dataset_name}',\n",
            "        'model': model_name,\n",
            "        'recall': m['recall'],\n",
            "        'precision': m['precision'],\n",
            "        'f1': m['f1'],\n",
            "        'f2': m['f2'],\n",
            "        'accuracy': m['accuracy'],\n",
            "        'pr_auc': m['pr_auc'],\n",
            "        'threshold': data['threshold']\n",
            "    })\n",
            "\n",
            "results_df = pd.DataFrame(results_list)\n",
            "\n",
            "# Display results\n",
            "print('\\n📋 Model Comparison:\\n')\n",
            "for _, row in results_df.iterrows():\n",
            "    print(f\"{row['model']}:\")\n",
            "    print(f\"   Recall:    {row['recall']:.4f} {'🎯' if row['recall'] >= 0.80 else ''}\")\n",
            "    print(f\"   Precision: {row['precision']:.4f}\")\n",
            "    print(f\"   F2 Score:  {row['f2']:.4f} {'⭐' if row['f2'] >= 0.75 else ''}\")\n",
            "    print(f\"   Accuracy:  {row['accuracy']:.4f}\")\n",
            "    print(f\"   Threshold: {row['threshold']:.2f}\\n\")\n",
            "\n",
            "# Export results\n",
            "csv_path = os.path.join(OUTPUT_DIR, f'results_{RUN_ID}.csv')\n",
            "json_path = os.path.join(OUTPUT_DIR, f'results_{RUN_ID}.json')\n",
            "\n",
            "results_df.to_csv(csv_path, index=False)\n",
            "results_df.to_json(json_path, orient='records', indent=2)\n",
            "\n",
            "print('\\n💾 Results saved to:')\n",
            "print(f'   CSV:  {csv_path}')\n",
            "print(f'   JSON: {json_path}')\n",
            "\n",
            "print('\\n' + '='*70)\n",
            f"print('✅ EXPERIMENT COMPLETE - {dataset_name}')\n",
            "print('='*70 + '\\n')\n",
            "\n",
            "# Display summary DataFrame\n",
            "print('\\n📊 Summary Table:')\n",
            "display(results_df[['model', 'recall', 'precision', 'f1', 'f2', 'accuracy', 'pr_auc']])\n"
        ]
    })

    # Create notebook structure
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return notebook


def main():
    """Main function to process all datasets."""

    datasets = [
        'CM1', 'JM1', 'KC1', 'KC2', 'KC3', 'KC4',
        'MC1', 'MC2', 'MW1', 'PC1', 'PC2', 'PC3', 'PC4'
    ]

    experiments_dir = Path('experiments')
    experiments_dir.mkdir(exist_ok=True)

    print('🚀 Restructuring NASA defect experiment notebooks...\n')

    for dataset in datasets:
        notebook = create_modular_notebook(dataset)
        output_path = experiments_dir / f'{dataset}_experiment.ipynb'

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2, ensure_ascii=False)

        print(f'✅ Created: {output_path}')

    print(f'\n🎉 All {len(datasets)} notebooks restructured successfully!')
    print('\n📝 Changes made:')
    print('   - Split large code blocks into modular cells')
    print('   - Each logical step is now a separate executable cell')
    print('   - Added clear section headers and documentation')
    print('   - Improved code organization and readability')
    print('\n💡 Notebooks are now ready to run cell-by-cell in Google Colab or Jupyter!')


if __name__ == '__main__':
    main()
