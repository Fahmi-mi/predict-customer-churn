import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
import joblib
import time

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Model imports
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# MLflow import
try:
    import mlflow
    import mlflow.sklearn
    import mlflow.lightgbm
    import mlflow.catboost
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Universal model trainer with cross-validation and MLflow tracking.
    
    Supports: LightGBM, CatBoost, XGBoost
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize trainer.
        
        Args:
            config: Configuration dictionary
            model_params: Custom model parameters (optional, will use defaults if None)
        """
        self.config = config
        self.model_config = config.get('model', {})
        
        self.model_type = self.model_config.get('type', 'lightgbm')
        self.task_type = self.model_config.get('task_type', 'classification')
        
        self.cv_config = self.model_config.get('cv', {})
        self.n_folds = self.cv_config.get('n_folds', 5)
        self.shuffle = self.cv_config.get('shuffle', True)
        self.stratified = self.cv_config.get('stratified', True)
        
        self.random_state = config.get('experiment', {}).get('seed', 42)
        self.model_params = model_params or self.model_config.get('params') or self._get_default_params()
        
        self.mlflow_config = config.get('mlflow', {})
        self.use_mlflow = self.mlflow_config.get('enabled', False) and MLFLOW_AVAILABLE
        
        self.models: List[Any] = []
        self.oof_predictions:Optional[np.ndarray] = None
        self.feature_importances:Optional[pd.DataFrame] = None
        self.y_train_encoded: Optional[pd.Series] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        logger.info(f"ModelTrainer initialized: {self.model_type}, task={self.task_type}")
        logger.info(f"Model params: {self.model_params}")
        
    def _get_default_params(self) -> Dict[str, Any]:
        """
        Get default parameters for selected model type.
        
        Returns:
            Default parameters dictionary
        """
        if self.model_type == 'lightgbm':
            return {
                'objective': 'binary' if self.task_type == 'classification' else 'regression',
                'metric': 'auc' if self.task_type == 'classification' else 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 20,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'verbose': -1,
                'random_state': self.random_state
            }
        elif self.model_type == 'catboost':
            return {
                'iterations': 1000,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'loss_function': 'Logloss' if self.task_type == 'classification' else 'RMSE',
                'eval_metric': 'AUC' if self.task_type == 'classification' else 'RMSE',
                'random_seed': self.random_state,
                'verbose': False,
                'early_stopping_rounds': 50
            }
        
        elif self.model_type == 'xgboost':
            return {
                'objective': 'binary:logistic' if self.task_type == 'classification' else 'reg:squarederror',
                'eval_metric': 'auc' if self.task_type == 'classification' else 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'seed': self.random_state
            }
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def train_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_test: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Train model with cross-validation.
        
        Args:
            X: Training features
            y: Training target
            X_test: Test features (optional, for prediction averaging)
        
        Returns:
            Tuple of (oof_predictions, test_predictions)
        """
        logger.info(f"Starting {self.n_folds}-fold cross-validation")
        logger.info(f"Training shape: {X.shape}")

        y_for_training = y.copy()
        if self.task_type == 'classification' and not pd.api.types.is_numeric_dtype(y_for_training):
            self.label_encoder = LabelEncoder()
            y_for_training = pd.Series(
                self.label_encoder.fit_transform(y_for_training.astype(str)),
                index=y_for_training.index,
                name=y_for_training.name
            )
            logger.info(f"Encoded target labels: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        self.y_train_encoded = y_for_training
        
        if self.use_mlflow:
            self._setup_mlflow()
            
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test)) if X_test is not None else None
        
        if self.task_type == 'classification' and self.stratified:
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            splits = kfold.split(X, y_for_training)
        else:
            kfold = KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            splits = kfold.split(X)
            
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold}/{self.n_folds}")
            logger.info(f"{'='*60}")
            
            start_time = time.time()
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y_for_training.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y_for_training.iloc[val_idx]
            
            model = self._train_single_model(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                fold
            )
            
            val_preds = self._predict(model, X_val_fold)
            oof_preds[val_idx] = val_preds
            
            if X_test is not None:
                test_preds += self._predict(model, X_test) / self.n_folds
                
            fold_score = self._calculate_metric(y_val_fold, val_preds)
            fold_scores.append(fold_score)
            
            elapsed = time.time() - start_time
            logger.info(f"Fold {fold} score: {fold_score:.6f} | Time: {elapsed:.2f} sec")
            
            if self.use_mlflow:
                mlflow.log_metric(f"fold_{fold}_score", fold_score)
                
            self.models.append(model)
            
        cv_score = self._calculate_metric(y_for_training, oof_preds)
        cv_std = np.std(fold_scores)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Cross-Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"CV Score: {cv_score:.6f} (+/- {cv_std:.6f})")
        logger.info(f"Fold scores: {fold_scores}")
        
        if self.use_mlflow:
            mlflow.log_metric("cv_score", cv_score)
            mlflow.log_metric("cv_std", cv_std)
            
            if self.mlflow_config.get('log_models', True):
                mlflow.sklearn.log_model(self.models[0], "model_fold_1")
                
        self.oof_predictions = oof_preds
        self._calculate_feature_importances(X.columns)
        
        return oof_preds, test_preds
    
    def _train_single_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        fold: int
    ) -> Any:
        """
        Train single model for one fold.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            fold: Fold number
        
        Returns:
            Trained model
        """
        if self.model_type == 'lightgbm':
            return self._train_lightgbm(X_train, y_train, X_val, y_val)
        elif self.model_type == 'catboost':
            return self._train_catboost(X_train, y_train, X_val, y_val)
        elif self.model_type == 'xgboost':
            return self._train_xgboost(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Any:
        """Train LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed. Install: pip install lightgbm")
        
        params = self.model_params.copy()
        params['n_jobs'] = self.config.get('performance', {}).get('parallel_jobs', -1)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100)
            ]
        )
        
        return model
    
    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Any:
        """Train CatBoost model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install: pip install catboost")
        
        params = self.model_params.copy()
        params['thread_count'] = self.config.get('performance', {}).get('parallel_jobs', -1)
        
        train_pool = cb.Pool(X_train, label=y_train)
        val_pool = cb.Pool(X_val, label=y_val)
        
        if self.task_type == 'classification':
            model = cb.CatBoostClassifier(**params)
        else:
            model = cb.CatBoostRegressor(**params)
            
        model.fit(train_pool, eval_set=val_pool, verbose=100)
        
        return model
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Any:
        """Train XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Install: pip install xgboost")
        
        params = self.model_params.copy()
        params['n_jobs'] = self.config.get('performance', {}).get('parallel_jobs', -1)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        evals = [(dtrain, 'train'), (dval, 'valid')]
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
    
    def _predict(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with model.
        
        Args:
            model: Trained model
            X: Features
        
        Returns:
            Predictions array
        """
        if self.model_type == 'lightgbm':
            return model.predict(X, num_iteration=model.best_iteration)
        
        elif self.model_type == 'catboost':
            if self.task_type == 'classification':
                preds = model.predict(X, prediction_type='Probability')[:, 1]
            else:
                preds = model.predict(X)
                
        elif self.model_type == 'xgboost':
            dtest = xgb.DMatrix(X)
            return model.predict(dtest, iteration_range=(0, model.best_iteration))
        
        else:
            preds = model.predict(X)
        
        return preds
            
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate evaluation metric.
        
        Args:
            y_true: True labels
            y_pred: Predictions
        
        Returns:
            Metric score
        """
        if self.task_type == 'classification':
            return roc_auc_score(y_true, y_pred)
        else:
            return np.sqrt(mean_squared_error(y_true, y_pred))
        
    def _calculate_feature_importances(self, feature_names: List[str]) -> None:
        """
        Calculate average feature importance across folds.
        
        Args:
            feature_names: List of feature names
        """
        importances = []
        
        for model in self.models:
            if self.model_type == 'lightgbm':
                imp = model.feature_importance(importance_type='gain')
                
            elif self.model_type == 'catboost':
                imp = model.get_feature_importance()
            
            elif self.model_type == 'xgboost':
                imp_dict = model.get_score(importance_type='gain')
                imp = [imp_dict.get(f'f{i}', 0) for i in range(len(feature_names))]
                
            else:
                continue
            
            importances.append(imp)
        
        avg_importance = np.mean(importances, axis=0)
        
        self.feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop 10 important features:")
        logger.info(self.feature_importances.head(10).to_string(index=False))
        
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if not self.use_mlflow:
            return
        
        tracking_uri = self.mlflow_config.get('tracking_uri', 'logs/mlruns')
        mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = self.mlflow_config.get('experiment_name', 'default')
        mlflow.set_experiment(experiment_name)
        
        run_name = self.mlflow_config.get('run_name', None)
        mlflow.start_run(run_name=run_name)
        
        if self.mlflow_config.get('log_params', True):
            mlflow.log_params(self.model_params)
            mlflow.log_param('model_type', self.model_type)
            mlflow.log_param('n_folds', self.n_folds)
            mlflow.log_param('task_type', self.task_type)
            
        tags = self.mlflow_config.get('tags', {})
        for key, value in tags.items():
            mlflow.set_tag(key, value)
            
        logger.info("MLflow tracking initialized")
        
    def save_models(self, output_dir: Optional[str] = None) -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Output directory (optional, will use config if not provided)
        """
        if output_dir is None:
            model_path_config = self.config.get('output', {}).get('model_path', 'experiments/default/model_final.pkl')
            output_dir = Path(model_path_config).parent
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models, 1):
            model_path = output_path / f"model_fold_{i}.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model fold {i} to: {model_path}")
        
        if self.feature_importances is not None:
            fi_csv_path = output_path / "feature_importance.csv"
            self.feature_importances.to_csv(fi_csv_path, index=False)
            logger.info(f"Saved feature importance CSV to: {fi_csv_path}")
            
            fi_plot_path = self.config.get('output', {}).get('feature_importance_path')
            if fi_plot_path:
                from src.evaluator import ModelEvaluator
                evaluator = ModelEvaluator(self.task_type)
                evaluator.plot_feature_importance(
                    self.feature_importances,
                    fi_plot_path,
                    top_n=20
                )
    
    def get_oof_predictions(self) -> np.ndarray:
        """Get out-of-fold predictions."""
        return self.oof_predictions
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances DataFrame."""
        return self.feature_importances