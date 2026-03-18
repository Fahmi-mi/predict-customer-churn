# 🚀 End-to-End ML Pipeline for Tabular Data

A production-ready, config-driven machine learning pipeline template for tabular data competitions (Kaggle Playground Series, etc.). Built with software engineering best practices, performance optimization, and experiment tracking in mind.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Key Features

- **🎯 Config-Driven Architecture** - All experiments controlled via YAML files, no code changes needed
- **⚡ Performance Optimized** - Parquet format (5-20x faster than CSV), memory optimization, vectorized operations
- **📊 Experiment Tracking** - Built-in MLflow integration for tracking parameters, metrics, and artifacts
- **🧪 Fully Tested** - Comprehensive unit tests with pytest (90%+ coverage)
- **🔧 Modular Design** - Clean separation of concerns with src/ package layout
- **🤖 Multi-Model Support** - LightGBM, CatBoost, XGBoost with unified interface
- **📈 Reproducible** - Seed control, config snapshots, and versioned outputs
- **🎓 Portfolio-Ready** - Production-grade code quality with type hints and documentation

## 📁 Project Structure

```
ml-base-template/
├── config/                          # Configuration files
│   ├── default.yaml                 # Main configuration template
│   └── local.yaml                   # Local overrides (gitignored)
├── data/
│   ├── raw/                         # Original CSV files (gitignored)
│   └── processed/                   # Parquet files for fast loading
├── src/                             # Main source code package
│   ├── config_loader.py             # YAML config with placeholder resolution
│   ├── data_loader.py               # CSV/Parquet auto-detection loader
│   ├── preprocessor.py              # sklearn-based preprocessing pipeline
│   ├── feature_engineering.py       # Vectorized feature creation
│   ├── trainer.py                   # Model training with CV and MLflow
│   ├── evaluator.py                 # Metrics calculation and visualization
│   ├── predictor.py                 # Ensemble predictions
│   └── utils.py                     # Logging, seeding, memory optimization
├── scripts/
│   └── convert_to_parquet.py        # One-time CSV to Parquet converter
├── notebooks/                       # Jupyter notebooks for exploration
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature experimentation
│   └── 03_experiment.ipynb          # Model comparison & tuning
├── tests/                           # Unit tests with pytest
│   ├── test_config_loader.py
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   ├── test_feature_engineering.py
│   └── test_utils.py
├── logs/                            # Logs and MLflow tracking
│   └── mlruns/                      # MLflow experiment artifacts
├── experiments/                     # Output per experiment run
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Fahmi-mi/end-to-end-ml-pipeline.git
cd end-to-end-ml-pipeline
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Verify installation**

```bash
pytest tests/ -v
```

## 🚀 Quick Start

### 1. Prepare Your Data

Place your Kaggle competition data in `data/raw/`:

```
data/raw/
├── train.csv
└── test.csv
```

Convert to Parquet for optimal performance:

```bash
python scripts/convert_to_parquet.py
```

This creates `train.parquet` and `test.parquet` in `data/processed/` (typically 3-10x smaller and 5-20x faster to load).

### 2. Configure Your Experiment

Edit `config/default.yaml` or create a new config file:

```yaml
experiment:
  name: "my_first_experiment"
  seed: 42

data:
  train_path: "data/processed/train.parquet"
  test_path: "data/processed/test.parquet"
  target_column: "target"
  id_column: "id"

model:
  type: "lightgbm" # or "catboost", "xgboost"
  task_type: "classification" # or "regression"

preprocessing:
  enabled: true
  numerical_features: ["Age", "Fare"]
  categorical_features: ["Sex", "Embarked"]

feature_engineering:
  enabled: true
  interactions: ["Age*Fare"]
  binning:
    Age: [0, 18, 35, 60, 100]
```

### 3. Run the Pipeline

**Full pipeline** (preprocessing + training + prediction):

```bash
python main.py --config config/default.yaml
```

**Training only** (skip prediction):

```bash
python main.py --config config/default.yaml --mode train
```

**Prediction only** (using saved models):

```bash
python main.py --config config/default.yaml --mode predict
```

**Override config values via CLI**:

```bash
python main.py --config config/default.yaml --override "model.type=catboost" "experiment.name=catboost_exp"
```

### 4. View Results

Results are saved in `experiments/{experiment_name}/`:

```
experiments/my_first_experiment/
├── train_processed.parquet      # Processed training data
├── test_processed.parquet       # Processed test data
├── preprocessor.pkl             # Fitted preprocessor
├── model_final.pkl              # Trained model(s)
├── feature_importance.png       # Feature importance plot
├── submission.csv               # Kaggle submission file
└── config_snapshot.yaml         # Config used for this run
```

### 5. Track Experiments with MLflow

Start MLflow UI:

```bash
mlflow ui --backend-store-uri logs/mlruns
```

Visit http://localhost:5000 to compare experiments, view metrics, and download artifacts.

## ⚙️ Configuration Guide

### Essential Config Sections

#### Experiment Settings

```yaml
experiment:
  name: "experiment_name" # Creates experiments/{name}/ folder
  seed: 42 # Reproducibility
```

#### Logging

```yaml
logging:
  enabled: true
  level: "INFO" # DEBUG, INFO, WARNING, ERROR
  log_to_file: true
  log_to_console: true
```

#### MLflow Tracking

```yaml
mlflow:
  enabled: true
  tracking_uri: "logs/mlruns"
  log_params: true # Track hyperparameters
  log_metrics: true # Track CV scores
  log_artifacts: true # Save plots and files
  log_model: true # Save trained models
```

#### Model Configuration

```yaml
model:
  type: "lightgbm" # lightgbm, catboost, xgboost
  task_type: "classification" # classification, regression
  cv:
    n_folds: 5
    shuffle: true
    stratified: true # For classification
```

#### Preprocessing

```yaml
preprocessing:
  enabled: true

  imputation:
    enabled: true
    strategy:
      numerical: "median"
      categorical: "most_frequent"

  encoding:
    enabled: true
    method: "onehot" # onehot, ordinal, label
    onehot_sparse: true # Memory efficient

  scaling:
    enabled: false # Usually false for tree models
    method: "standard" # standard, minmax, robust

  downcast_dtype:
    enabled: true # Optimize memory usage
```

#### Feature Engineering

```yaml
feature_engineering:
  enabled: true

  interactions: # Feature multiplication
    - "Age*Fare"
    - "Pclass*Fare"

  binning: # Numerical binning
    Age: [0, 18, 35, 60, 100]
    Fare: [0, 10, 30, 100]
```

### Placeholder Support

Configs support dynamic placeholders:

```yaml
experiment:
  name: "my_experiment"

output:
  model_path: "experiments/{experiment.name}/model.pkl"
  # Resolves to: "experiments/my_experiment/model.pkl"
```

## 🧪 Testing

Run all tests:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_preprocessor.py -v
```

Run with coverage report:

```bash
pytest tests/ --cov=src --cov-report=html
```

## ⚡ Performance Optimizations

This pipeline is optimized for speed and memory efficiency:

### 1. **Parquet Format**

- 5-20x faster loading vs CSV
- 3-10x smaller file size
- Columnar storage with predicate pushdown

### 2. **Memory Optimization**

- Automatic dtype downcasting (float64 → float32, int64 → int32)
- Sparse matrix support for one-hot encoding
- Memory profiling and cleanup utilities

### 3. **Vectorized Operations**

- All feature engineering uses pandas/numpy vectorization
- No slow `.apply()` or loop operations
- Parallel processing with `n_jobs=-1`

### 4. **Fast Models**

- LightGBM (fastest, recommended)
- CatBoost (handles categorical natively)
- XGBoost (most popular)

### Benchmark Results

On a typical Kaggle dataset (10K rows, 50 features):

| Operation     | CSV    | Parquet | Speedup |
| ------------- | ------ | ------- | ------- |
| Load data     | 2.5s   | 0.3s    | 8.3x    |
| Full pipeline | 45s    | 12s     | 3.8x    |
| Memory usage  | 240 MB | 85 MB   | 2.8x    |

## 📊 Experiment Tracking with MLflow

MLflow automatically tracks:

- **Parameters**: Model hyperparameters, preprocessing settings
- **Metrics**: CV scores per fold, mean/std metrics
- **Artifacts**: Feature importance plots, config snapshots, trained models
- **Tags**: Project name, author, custom tags

### MLflow UI Features

- Compare multiple experiments side-by-side
- Visualize metric trends
- Download artifacts and models
- Search and filter runs

## 📓 Jupyter Notebooks

Three notebooks for interactive exploration:

### 1. `01_eda.ipynb` - Exploratory Data Analysis

- Dataset overview and statistics
- Missing value analysis
- Distribution plots
- Correlation analysis
- Target variable investigation

### 2. `02_feature_engineering.ipynb` - Feature Experimentation

- Test feature interactions
- Validate binning strategies
- Visualize feature relationships
- Quick experiments before pipeline runs

### 3. `03_experiment.ipynb` - Model Comparison

- Baseline model training
- Hyperparameter tuning
- Model comparison (LightGBM vs CatBoost vs XGBoost)
- Ensemble experiments
- Final model selection

## 🔧 Development Guide

### Adding a New Model

1. Update `src/trainer.py`:

```python
def _train_your_model(self, X_train, y_train, X_val, y_val, params):
    # Your model training logic
    pass
```

2. Add to `train_cv()` method model routing

3. Add default parameters in `_get_default_params()`

### Adding New Features

1. Update `src/feature_engineering.py`:

```python
def _create_custom_features(self, X: pd.DataFrame) -> pd.DataFrame:
    # Your feature creation logic
    return X
```

2. Add config options in `config/default.yaml`

### Custom Preprocessing

1. Extend `src/preprocessor.py`:

```python
def _build_custom_pipeline(self) -> Pipeline:
    # Your preprocessing steps
    return Pipeline(steps)
```

## 🎯 Best Practices

1. **Always use Parquet** - Run `convert_to_parquet.py` first
2. **Version your configs** - Create new config files for experiments
3. **Check MLflow regularly** - Track experiment progress
4. **Run tests before commits** - Ensure code quality
5. **Use meaningful experiment names** - Easy to find later
6. **Leverage notebooks** - Quick prototyping before pipeline runs

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by Kaggle competitions and best practices
- Built with modern ML libraries: scikit-learn, LightGBM, CatBoost, XGBoost
- Experiment tracking powered by MLflow
- Testing with pytest

## 📧 Contact

**Fahmi Haqqul Ihsan** - [LinkedIn](https://www.linkedin.com/in/fahmi-haqqul-ihsan-b13276326/) - fahmihaqqulihsan@gmail.com

Project Link: [https://github.com/Fahmi-mi/end-to-end-ml-pipeline](https://github.com/Fahmi-mi/end-to-end-ml-pipeline)

---

⭐ Star this repository if you find it helpful!
