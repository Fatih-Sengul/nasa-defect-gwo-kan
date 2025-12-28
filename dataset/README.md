# NASA MDP Dataset Directory

## 📁 Place Your NASA MDP Datasets Here

This directory should contain NASA MDP (Metrics Data Program) datasets in `.arff` format.

### Supported Datasets

- **CM1** - NASA spacecraft instrument (C language)
- **JM1** - Real-time predictive ground system (C language)
- **KC1** - Storage management for ground data (C++)
- **KC2** - Science data processing (C++)
- **KC3** - Flight software for space mission (Java)
- **MC2** - Video guidance system (C++)
- **PC1** - Flight software for earth orbiting satellite (C)
- **MW1** - Zero defect NASA software (C)

### Dataset Sources

You can obtain NASA MDP datasets from:

1. **PROMISE Repository**: http://promise.site.uottawa.ca/SERepository/
2. **OpenML**: https://www.openml.org/
3. **NASA MDP Official**: https://nasa-public-data.s3.amazonaws.com/midas/

### File Format

Files should be in ARFF (Attribute-Relation File Format):

```
dataset/
├── CM1.arff
├── JM1.arff
├── KC1.arff
├── KC2.arff
├── MC2.arff
└── PC1.arff
```

### Dataset Structure

Each dataset typically contains:

- **Features**: Software metrics (LOC, Cyclomatic Complexity, Halstead metrics, etc.)
- **Target**: Defective (True/False or Y/N)

### Quick Download Example

```bash
# Example using wget to download from PROMISE repository
cd dataset/

# Download CM1
wget http://promise.site.uottawa.ca/SERepository/datasets/cm1.arff -O CM1.arff

# Download JM1
wget http://promise.site.uottawa.ca/SERepository/datasets/jm1.arff -O JM1.arff

# Repeat for other datasets...
```

### Validation

After placing datasets, verify they're detected:

```python
from pathlib import Path

dataset_dir = Path('./dataset/')
arff_files = list(dataset_dir.glob('*.arff'))
print(f"Found {len(arff_files)} datasets: {[f.name for f in arff_files]}")
```

### Notes

- The pipeline automatically processes all `.arff` files in this directory
- Ensure datasets are properly formatted (ARFF format)
- Remove any corrupted or incomplete files before running the pipeline
- The framework handles byte-string decoding and preprocessing automatically

---

**Once datasets are in place, run `main_gwo_kan.ipynb` to begin the analysis!**
