import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import List, Optional, Dict, Any
import logging

from src.utils import reduce_mem_usage

logger = logging.getLogger(__name__)

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Complete preprocessing pipeline for tabular data.
    
    Handles:
    - Missing value imputation
    - Categorical encoding (OneHot, Ordinal, Label)
    - Numerical scaling
    - Memory optimization (downcast dtypes)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with config.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config.get('preprocessing', {})
        self.numerical_features = self.preprocessing_config.get('numerical_features', [])
        self.categorical_features = self.preprocessing_config.get('categorical_features', [])
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_out: Optional[List[str]] = None
        
        self.is_fitted = False
        
        logger.info("DataPreprocessor initialized")
        
    def fit(self, X:pd.DataFrame, y:Optional[pd.Series]=None) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Training features
            y: Training target (optional, not used)
        
        Returns:
            Self
        """
        if not self.preprocessing_config.get('enabled', True):
            logger.info("Preprocessing disabled, skipping fit")
            self.is_fitted = True
            return self
        
        logger.info("Fitting preprocessor....")
        
        if not self.numerical_features and not self.categorical_features:
            self._auto_detect_features(X)
            
        self.preprocessor = self._build_pipeline()
        self.preprocessor.fit(X)
        self._set_feature_names()
        
        self.is_fitted = True
        logger.info(f"Preprocessor fitted. Output features: {len(self.feature_names_out)}")
        
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")
        
        if not self.preprocessing_config.get('enabled', True):
            logger.info("Preprocessing disabled, returning original data")
            return X.copy()
        
        X_transformed = self.preprocessor.transform(X)
        
        df_transformed = pd.DataFrame(
            X_transformed,
            columns=self.feature_names_out,
            index=X.index
        )
        
        if self.preprocessing_config.get('downcast_dtype', {}).get('enabled', True):
            df_transformed = reduce_mem_usage(df_transformed, verbose=False)
            
        logger.info(f"Transformation complete. Output shape: {df_transformed.shape}")
        
        return df_transformed
    
    def fit_transform(self, X:pd.DataFrame, y:Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Features
            y: Target (optional)
        
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _auto_detect_features(self, X: pd.DataFrame) -> None:
        """
        Auto-detect numerical and categorical features.
        
        Args:
            X: Input DataFrame
        """
        self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Auto-detected {len(self.numerical_features)} numerical features")
        logger.info(f"Auto-detected {len(self.categorical_features)} categorical features")
    
    def _build_pipeline(self) -> ColumnTransformer:
        """
        Build preprocessing pipeline based on config.
        
        Returns:
            ColumnTransformer pipeline
        """
        transformers = []
        
        if self.numerical_features:
            num_pipeline = self._build_numerical_pipeline()
            transformers.append(('num', num_pipeline, self.numerical_features))
            
        if self.categorical_features:
            cat_pipeline = self._build_categorical_pipeline()
            transformers.append(('cat', cat_pipeline, self.categorical_features))
            
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            sparse_threshold=0.3,
            n_jobs=self.config.get('performance', {}).get('parallel_jobs', -1)
        )
        
        return preprocessor
    
    def _build_numerical_pipeline(self) -> Pipeline:
        """
        Build numerical feature pipeline.
        
        Returns:
            Numerical pipeline
        """
        steps = []
        
        if self.preprocessing_config.get('imputation', {}).get('enabled', True):
            strategy = self.preprocessing_config.get('imputation', {}).get('strategy', {}).get('numerical', 'median')
            imputer = SimpleImputer(strategy=strategy)
            steps.append(('imputer', imputer))
            
        if self.preprocessing_config.get('scaling', {}).get('enabled', False):
            method = self.preprocessing_config.get('scaling', {}).get('method', 'standard')
            apply_to = self.preprocessing_config.get('scaling', {}).get('apply_to', 'numerical_only')
            
            if apply_to in ['numerical_only', 'all']:
                scaler = self._get_scaler(method)
                steps.append(('scaler', scaler))
                
        return Pipeline(steps) if steps else Pipeline([('passthrough', 'passthrough')])
    
    def _build_categorical_pipeline(self) -> Pipeline:
        """
        Build categorical feature pipeline.
        
        Returns:
            Categorical pipeline
        """
        steps = []
        
        if self.preprocessing_config.get('imputation', {}).get('enabled', True):
            strategy = self.preprocessing_config.get('imputation', {}).get('strategy', {}).get('categorical', 'most_frequent')
            imputer = SimpleImputer(strategy=strategy, fill_value='missing')
            steps.append(('imputer', imputer))
            
        if self.preprocessing_config.get('encoding', {}).get('enabled', True):
            method = self.preprocessing_config.get('encoding', {}).get('method', 'onehot')
            encoder = self._get_encoder(method)
            steps.append(('encoder', encoder))
            
        return Pipeline(steps) if steps else Pipeline([('passthrough', 'passthrough')])
    
    def _get_scaler(self, method:str):
        """
        Get scaler instance by method name.
        
        Args:
            method: Scaler method ('standard', 'minmax', 'robust')
        
        Returns:
            Scaler instance
        """
        scaler = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        
        if method not in scaler:
            logger.warning(f"Unknown scaling method: {method}, using 'standard'")
            method = 'standard'
            
        return scaler[method]
    
    def _get_encoder(self, method: str):
        """
        Get encoder instance by method name.
        
        Args:
            method: Encoder method ('onehot', 'ordinal', 'label')
        
        Returns:
            Encoder instance
        """
        if method == 'onehot':
            sparse = self.preprocessing_config.get('encoding', {}).get('onehot_sparse', True)
            return OneHotEncoder(
                sparse_output=sparse,
                handle_unknown='ignore',
                drop=None
            )
            
        elif method == 'ordinal':
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        elif method == 'label':
            logger.warning("LabelEncoder not compatible with pipeline, using OrdinalEncoder")
            return OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        
        else:
            logger.warning(f"Unknown encoding method: {method}, using 'onehot'")
            return OneHotEncoder(
                sparse_output=True,
                handle_unknown='ignore',
                drop=None
            )
            
    def _set_feature_names(self) -> None:
        """
        Extract and store output feature names from fitted pipeline.
        """
        try:
            self.feature_names_out = self.preprocessor.get_feature_names_out().tolist()
        except AttributeError:
            self.feature_names_out = self._construct_feature_names()
            
        self.feature_names_out = [
            name.replace('num__', '').replace('cat__', '').replace('remainder__', '') for name in self.feature_names_out
        ]
        
    def _construct_feature_names(self) -> List[str]:
        """
        Manually construct output feature names.
        
        Returns:
            List of feature names
        """
        feature_names = []
        
        feature_names.extend(self.numerical_features)
        
        encoding_method = self.preprocessing_config.get('encoding', {}).get('method', 'onehot')
        
        if encoding_method == 'onehot':
            for cat_feat in self.categorical_features:
                feature_names.append(cat_feat)
        else:
            feature_names.extend(self.categorical_features)
            
        return feature_names
    
    def get_feature_names(self) -> List[str]:
        """
        Get output feature names.
        
        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted first")
        
        return self.feature_names_out
    
def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: Dict[str, Any],
    y_train: Optional[pd.Series] = None,
) -> tuple:
    """
    Convenience function to preprocess train and test data.

    Args:
        X_train: Training features
        X_test: Test features
        config: Configuration dictionary
        y_train: Training target (optional)

    Returns:
        Tuple of (X_train_processed, X_test_processed, preprocessor)
    """
    preprocessor = DataPreprocessor(config)
    
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_test_processed, preprocessor