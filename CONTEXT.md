# KONTEKS PROYEK UNTUK GITHUB COPILOT

# Tabular ML Pipeline - Portofolio Hybrid Software Engineer + AI/ML

# Updated: January 01, 2026 (v2.0 - Added Logging & MLflow)

## Tujuan Utama Proyek

Ini adalah portofolio pribadi yang menonjolkan kemampuan sebagai Software Engineer yang masuk ke dunia AI/ML. Fokus utama:

- Kode bersih, modular, dan mudah dipelihara (software engineering best practices)
- Pipeline end-to-end untuk data tabular (Kaggle Playground / kompetisi)
- Config-driven menggunakan YAML (mudah eksperimen tanpa ubah kode)
- Performa komputasi tinggi (bukan hanya akurasi model)
- Reproducible, testable, dan scalable

## Prioritas Teknis yang Sudah Diputuskan

1. **Format data**: Gunakan Parquet (snappy compression) di folder data/processed/

   - Ada script terpisah: scripts/convert_to_parquet.py untuk konversi sekali dari CSV

2. **Data loading**:

   - src/data_loader.py harus support otomatis CSV atau Parquet berdasarkan ekstensi file

3. **Optimasi performa komputasi**:

   - Downcast dtype otomatis (float64 → float32, int64 → int32)
   - OneHotEncoder dengan sparse=True
   - Semua operasi di feature engineering HARUS vectorized (Pandas/NumPy)
   - Parallel processing: n_jobs=-1
   - Prioritas model cepat: LightGBM > CatBoost > XGBoost

4. **Code quality**:

   - Wajib pakai type hints bawaan Python
   - Style: PEP8, docstrings jelas
   - Belum pakai linter eksternal (Ruff/Black) tapi siap ditambah nanti

5. **Logging & Experiment Tracking**:

   - Logging: comprehensive logging dengan level configurability (DEBUG/INFO/WARNING/ERROR)
   - MLflow: tracking parameters, metrics, artifacts, dan models
   - Log location: logs/ folder dengan mlruns/ subdirectory untuk MLflow
   - Integration: setup_logger() di utils.py, MLflow logging di trainer.py

6. **Notebooks**:

   - Hanya 3 notebook utama:
     - 01_eda.ipynb → EDA + data checking
     - 02_feature_engineering.ipynb → Eksperimen fitur + validasi visual
     - 03_experiment.ipynb → Baseline, tuning, model comparison, ensemble (bebas!)
   - Folder archive/ untuk versi lama

7. **Testing**:

   - Unit tests dengan pytest di folder tests/
   - Sudah direncanakan: test_config_loader, test_data_loader (termasuk Parquet), test_preprocessor, test_feature_engineering, test_utils

8. **Struktur yang sudah final**:
   - Semua config di folder config/
   - local.yaml untuk override pribadi (di-.gitignore)
   - src/ sebagai Python package (dengan **init**.py)
   - logs/ untuk log files dan mlruns/ (MLflow tracking)
   - experiments/ untuk output per run
   - main.py sebagai entry point dengan argparse dan timing total execution

## Gaya Coding yang Diharapkan

- Modular: setiap tahap pipeline di file terpisah di src/
- Configurable: hampir semua hal diatur lewat YAML
- Defensive: error handling yang baik, logging informatif
- Performant: hindari operasi lambat, prioritaskan vectorized dan memory-efficient
- Readable: docstring, type hints, comment secukupnya

## Default Config Structure (config/default.yaml)

Ini adalah struktur lengkap config/default.yaml yang harus diikuti. Semua path di output mendukung placeholder `{experiment.name}` yang akan diganti otomatis.

```yaml
experiment:
  name: "default_preprocessing" # Nama folder di experiments/
  seed: 42 # Untuk reproducibility

logging:
  enabled: true
  level: "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_to_file: true
  log_to_console: true
  log_dir: "logs"
  log_filename: "{experiment.name}.log" # Support placeholder

mlflow:
  enabled: true
  tracking_uri: "logs/mlruns" # Path ke MLflow tracking directory
  experiment_name: "{experiment.name}" # Auto dari experiment.name
  run_name: null # null = auto-generate dengan timestamp
  log_params: true # Log semua hyperparameters
  log_metrics: true # Log metrics per CV fold
  log_artifacts: true # Log plots dan files
  log_model: true # Log trained model
  tags:
    project: "ml-base-template"
    author: "Your Name"

model:
  type: "lightgbm" # Model type: "lightgbm", "catboost", "xgboost"
  task_type: "classification" # Task type: "classification" atau "regression"

  cv:
    n_folds: 5 # Jumlah fold untuk cross-validation
    shuffle: true # Shuffle data sebelum split
    stratified: true # Stratified split (true untuk classification, false untuk regression)

  params: # Hyperparameters model (sesuaikan dengan model.type)
    # LightGBM default params
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    max_depth: -1
    min_child_samples: 20
    reg_alpha: 0.0
    reg_lambda: 0.0

    # CatBoost params (uncomment jika type: "catboost")
    # depth: 6
    # l2_leaf_reg: 3
    # border_count: 128
    # iterations: 1000

    # XGBoost params (uncomment jika type: "xgboost")
    # max_depth: 6
    # min_child_weight: 1
    # gamma: 0
    # subsample: 0.8
    # colsample_bytree: 0.8

data:
  train_path: "data/processed/train.parquet"
  test_path: "data/processed/test.parquet"
  target_column: "target" # Ganti sesuai dataset
  id_column: "id" # Untuk submission (kosongkan jika tidak ada)
  drop_columns: [] # Kolom yang langsung dibuang

preprocessing:
  enabled: true # Master switch untuk skip semua preprocessing

  numerical_features: [] # Contoh: ["Age", "Fare", "SibSp", "Parch"]
  categorical_features: [] # Contoh: ["Sex", "Embarked", "Pclass"]

  imputation:
    enabled: true # false → skip imputation
    strategy:
      numerical: "median" # "mean", "median", "constant"
      categorical: "most_frequent" # "most_frequent", "constant"

  encoding:
    enabled: true # false → biarkan categorical (bagus untuk CatBoost)
    method: "onehot" # "onehot", "ordinal", "label"
    onehot_sparse: true # Hemat memori

  scaling:
    enabled: false # Default false karena tree-based tidak butuh
    method: "standard" # "standard", "minmax", "robust"
    apply_to: "numerical_only" # atau "all"

  downcast_dtype:
    enabled: true # Hampir selalu true → hemat memori

feature_engineering:
  enabled: true

  interactions: [] # Contoh: ["Age*Fare", "Pclass*Fare"]

  binning:
    {} # Contoh:
    # Age: [0, 12, 18, 35, 60, 100]
    # Fare: [0, 8, 15, 31, 1000]

performance:
  parallel_jobs: -1 # n_jobs untuk paralelisme
  cache_preprocessor: false # Nanti bisa true + joblib

output:
  processed_train_path: "experiments/{experiment.name}/train_processed.parquet"
  processed_test_path: "experiments/{experiment.name}/test_processed.parquet"
  preprocessor_path: "experiments/{experiment.name}/preprocessor.pkl" # Optional cache

  # Tempat hasil modeling final (diisi manual dari notebook)
  model_path: "experiments/{experiment.name}/model_final.pkl"
  submission_path: "experiments/{experiment.name}/submission.csv"
  feature_importance_path: "experiments/{experiment.name}/feature_importance.png"
```
