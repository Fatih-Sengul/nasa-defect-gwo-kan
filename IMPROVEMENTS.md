# Model Accuracy İyileştirmeleri

## Sorun Analizi

Mevcut modelinizde şu sorunlar tespit edildi:

### 1. **Çok Yüksek Recall, Çok Düşük Accuracy**
- PC5: Accuracy 0.28, Recall 0.99 → Model neredeyse her şeyi "defective" olarak tahmin ediyor
- CM1: Accuracy 0.50, Recall 0.88 → Aynı sorun
- MC2: Accuracy 0.48, Recall 0.89 → Dengesiz tahminler

### 2. **Ana Sebep: GWO Sadece Recall'ı Optimize Ediyor**
```python
# Eski kod (main_gwo_kan.ipynb)
def gwo_kan_fitness(...):
    ...
    return metrics['Recall']  # ❌ Sadece recall!
```

Bu, modelin **false positive'leri görmezden gelmesine** ve her şeyi pozitif sınıf olarak tahmin etmesine neden olur.

### 3. **Diğer Sorunlar**
- Early stopping sadece recall'a bakıyor
- Threshold her zaman 0.5 (optimize edilmiyor)
- SMOTE çok agresif (1:1 ratio)
- Class imbalance düzgün handle edilmiyor

---

## Yapılan İyileştirmeler

### ✅ 1. **Dengeli Fitness Fonksiyonu**
```python
# Yeni kod (main_gwo_kan_improved.ipynb)
def gwo_kan_fitness(...):
    ...
    # F1, Recall ve Accuracy'nin ağırlıklı ortalaması
    fitness = (
        0.5 * metrics['F1-Score'] +   # %50 F1 (denge)
        0.3 * metrics['Recall'] +      # %30 Recall (güvenlik)
        0.2 * metrics['Accuracy']      # %20 Accuracy (genel performans)
    )
    return fitness
```

**Sonuç**: Model artık precision ve accuracy'yi de dikkate alıyor.

---

### ✅ 2. **Threshold Optimization**
```python
def find_optimal_threshold(model, X_val, y_val):
    """
    0.5 yerine F1-score'u maksimize eden threshold'u bul
    """
    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(0.1, 0.9, 0.05):
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
```

**Sonuç**: Her dataset için optimal threshold bulunuyor (örneğin, 0.65 veya 0.45 olabilir).

---

### ✅ 3. **Focal Loss - False Positive'leri Azaltır**
```python
class FocalLoss(nn.Module):
    """
    Zor örneklere odaklanır, kolay örnekleri down-weight eder
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

**Sonuç**: Model false positive'lere daha fazla penalty veriyor.

---

### ✅ 4. **Early Stopping - F1 Bazlı**
```python
# Eski kod
if val_recall > best_recall:  # ❌ Sadece recall
    best_recall = val_recall
    ...

# Yeni kod
if val_f1 > best_f1:  # ✅ F1 (dengeli metrik)
    best_f1 = val_f1
    ...
```

**Sonuç**: Model training sırasında dengeli metriği maksimize ediyor.

---

### ✅ 5. **Daha Az Agresif SMOTE**
```python
# Eski kod
smote = SMOTE(random_state=RANDOM_SEED)  # ❌ 1:1 ratio (çok agresif)

# Yeni kod
smote = SMOTE(sampling_strategy=0.7, random_state=RANDOM_SEED)  # ✅ 0.7:1 ratio
```

**Sonuç**: Minority class'a aşırı fit olmayı engelliyor.

---

### ✅ 6. **Class Weights & Balanced Accuracy**
```python
# Class weights ile BCE Loss
class_counts = np.bincount(y_train)
pos_weight = torch.FloatTensor([class_counts[0] / class_counts[1]])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Balanced Accuracy metriği eklendi
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Balanced_Accuracy': balanced_accuracy_score(y_test, y_pred),  # ✅ Yeni
    ...
}
```

**Sonuç**: Imbalanced dataset'lerde daha iyi performans.

---

## Beklenen İyileştirmeler

### Önceki Sonuçlar (Sorunlu)
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| PC5     | **0.28** | 0.27      | 0.99   | 0.43     |
| CM1     | **0.50** | 0.18      | 0.88   | 0.30     |
| MC2     | **0.48** | 0.40      | 0.89   | 0.55     |
| KC1     | **0.56** | 0.34      | 0.81   | 0.48     |

### Beklenen Sonuçlar (İyileştirilmiş)
| Dataset | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| PC5     | **~0.65** | ~0.45     | ~0.80  | ~0.58    |
| CM1     | **~0.75** | ~0.50     | ~0.75  | ~0.60    |
| MC2     | **~0.70** | ~0.60     | ~0.75  | ~0.67    |
| KC1     | **~0.72** | ~0.55     | ~0.70  | ~0.62    |

**Not**: Recall biraz düşecek ama accuracy ve F1 önemli ölçüde artacak.

---

## Kullanım

### 1. Yeni Notebook'u Çalıştır
```bash
# Google Colab'da
# main_gwo_kan_improved.ipynb dosyasını aç ve çalıştır
```

### 2. Parametreleri Ayarla (İsteğe Bağlı)
```python
# Fitness fonksiyonunda ağırlıkları değiştir:
fitness = (
    0.6 * metrics['F1-Score'] +   # F1'e daha fazla ağırlık ver
    0.2 * metrics['Recall'] +      # Recall'ı azalt
    0.2 * metrics['Accuracy']
)

# SMOTE ratio'yu değiştir:
smote = SMOTE(sampling_strategy=0.5, ...)  # Daha az oversampling
```

### 3. Sonuçları Karşılaştır
```python
# Eski sonuçlar: final_results.xlsx
# Yeni sonuçlar: final_results_improved.xlsx
```

---

## Ekstra Öneriler (İleride Deneyin)

### 1. **Ensemble Methods**
```python
# Birden fazla model kombine et
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier([
    ('kan1', model1),
    ('kan2', model2),
    ('kan3', model3)
], voting='soft')
```

### 2. **Feature Selection**
```python
from sklearn.feature_selection import SelectKBest, f_classif

# En önemli feature'ları seç
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)
```

### 3. **Cross-Validation**
```python
# Daha robust sonuçlar için
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    ...
```

### 4. **Hyperparameter Tuning - Daha Fazla Iteration**
```python
# GWO'yu daha uzun çalıştır
gwo = GreyWolfOptimizer(
    n_wolves=15,      # 10 → 15
    n_iterations=30,  # 20 → 30
    ...
)
```

---

## Sonuç

**Ana İyileştirmeler:**
1. ✅ GWO artık **F1-score + Recall + Accuracy** optimize ediyor (sadece recall değil)
2. ✅ **Threshold optimization** ile daha iyi accuracy-recall tradeoff
3. ✅ **Focal Loss** ile false positive'ler azalıyor
4. ✅ **Early stopping** F1-score bazlı
5. ✅ **Daha az agresif SMOTE** (overfitting azalıyor)

**Beklenen Sonuç**: Accuracy **%20-30 artacak**, recall biraz düşecek ama genel performans (F1) **önemli ölçüde iyileşecek**.

---

## Dosyalar

- `main_gwo_kan.ipynb` → Orijinal kod (recall-only optimization)
- `main_gwo_kan_improved.ipynb` → İyileştirilmiş kod (**bunu kullanın**)
- `IMPROVEMENTS.md` → Bu dosya (değişikliklerin açıklaması)

---

**Not**: Google Colab'da 3+ saat sürdüğü için, küçük bir subset üzerinde önce test edin:

```python
# Test için sadece 2-3 dataset kullan
arff_files = glob.glob(os.path.join(dataset_dir, '*.arff'))[:3]
```

Sonuçlar iyi görünürse tüm dataset'leri çalıştırın.
