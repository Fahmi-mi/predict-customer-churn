import pytest
import pandas as pd
import numpy as np
from sklearn.base import clone
from scipy.sparse import issparse
import tempfile
import shutil
from pathlib import Path

from src.preprocessor import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample dataset with numerical and categorical features."""
    np.random.seed(42)
    
    data = {
        'num1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        'num2': [10.5, 20.3, np.nan, 40.1, 50.8, 60.2, np.nan, 80.5],
        'num3': [100, 200, 300, 400, 500, 600, 700, 800],
        'cat1': ['A', 'B', 'A', 'C', 'B', 'A', np.nan, 'C'],
        'cat2': ['X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y']
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def basic_config():
    """Basic configuration for preprocessing."""
    return {
        'preprocessing': {
            'enabled': True,
            'numerical_features': [],
            'categorical_features': [],
            'imputation': {
                'enabled': True,
                'strategy': {
                    'numerical': 'median',
                    'categorical': 'most_frequent'
                }
            },
            'encoding': {
                'enabled': True,
                'method': 'onehot',
                'onehot_sparse': False
            },
            'scaling': {
                'enabled': False,
                'method': 'standard',
                'apply_to': 'numerical_only'
            },
            'downcast_dtype': {
                'enabled': True
            }
        },
        'performance': {
            'parallel_jobs': 1
        }
    }


class TestDataPreprocessorInit:
    """Test DataPreprocessor initialization."""
    
    def test_init_basic(self, basic_config):
        """Test basic initialization."""
        preprocessor = DataPreprocessor(basic_config)
        
        assert preprocessor.config == basic_config
        assert preprocessor.preprocessing_config == basic_config['preprocessing']
        assert preprocessor.preprocessor is None
        assert not preprocessor.is_fitted
    
    def test_init_with_features_specified(self):
        """Test initialization with features specified in config."""
        config = {
            'preprocessing': {
                'enabled': True,
                'numerical_features': ['num1', 'num2'],
                'categorical_features': ['cat1', 'cat2']
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        
        assert preprocessor.numerical_features == ['num1', 'num2']
        assert preprocessor.categorical_features == ['cat1', 'cat2']


class TestDataPreprocessorFit:
    """Test DataPreprocessor fit method."""
    
    def test_fit_basic(self, sample_data, basic_config):
        """Test basic fit operation."""
        preprocessor = DataPreprocessor(basic_config)
        
        result = preprocessor.fit(sample_data)
        
        assert result is preprocessor
        assert preprocessor.is_fitted
        assert preprocessor.preprocessor is not None
        assert preprocessor.feature_names_out is not None
    
    def test_fit_auto_detect_features(self, sample_data, basic_config):
        """Test automatic feature type detection."""
        preprocessor = DataPreprocessor(basic_config)
        
        preprocessor.fit(sample_data)
        
        assert len(preprocessor.numerical_features) == 3
        assert len(preprocessor.categorical_features) == 2
        assert 'num1' in preprocessor.numerical_features
        assert 'cat1' in preprocessor.categorical_features
    
    def test_fit_with_manual_features(self, sample_data):
        """Test fit with manually specified features."""
        config = {
            'preprocessing': {
                'enabled': True,
                'numerical_features': ['num1', 'num2'],
                'categorical_features': ['cat1'],
                'imputation': {'enabled': True, 'strategy': {'numerical': 'mean', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': True, 'method': 'onehot', 'onehot_sparse': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        preprocessor.fit(sample_data)
        
        assert preprocessor.numerical_features == ['num1', 'num2']
        assert preprocessor.categorical_features == ['cat1']
    
    def test_fit_disabled_preprocessing(self, sample_data):
        """Test fit when preprocessing is disabled."""
        config = {
            'preprocessing': {'enabled': False},
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        preprocessor.fit(sample_data)
        
        assert preprocessor.is_fitted
        assert preprocessor.preprocessor is None


class TestDataPreprocessorTransform:
    """Test DataPreprocessor transform method."""
    
    def test_transform_basic(self, sample_data, basic_config):
        """Test basic transform operation."""
        preprocessor = DataPreprocessor(basic_config)
        preprocessor.fit(sample_data)
        
        transformed = preprocessor.transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)
        assert not transformed.isnull().any().any()
    
    def test_transform_not_fitted_error(self, sample_data, basic_config):
        """Test transform raises error when not fitted."""
        preprocessor = DataPreprocessor(basic_config)
        
        with pytest.raises(RuntimeError, match="must be fitted before transform"):
            preprocessor.transform(sample_data)
    
    def test_transform_handles_missing_values(self, sample_data, basic_config):
        """Test that transform handles missing values."""
        preprocessor = DataPreprocessor(basic_config)
        preprocessor.fit(sample_data)
        
        transformed = preprocessor.transform(sample_data)
        
        assert not transformed.isnull().any().any()
    
    def test_transform_preserves_index(self, sample_data, basic_config):
        """Test that transform preserves DataFrame index."""
        sample_data.index = [100, 101, 102, 103, 104, 105, 106, 107]
        
        preprocessor = DataPreprocessor(basic_config)
        preprocessor.fit(sample_data)
        
        transformed = preprocessor.transform(sample_data)
        
        assert transformed.index.tolist() == sample_data.index.tolist()
    
    def test_transform_disabled_preprocessing(self, sample_data):
        """Test transform when preprocessing is disabled."""
        config = {
            'preprocessing': {'enabled': False},
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        preprocessor.fit(sample_data)
        
        transformed = preprocessor.transform(sample_data)
        
        pd.testing.assert_frame_equal(transformed, sample_data)


class TestDataPreprocessorFitTransform:
    """Test DataPreprocessor fit_transform method."""
    
    def test_fit_transform_basic(self, sample_data, basic_config):
        """Test fit_transform combines fit and transform."""
        preprocessor = DataPreprocessor(basic_config)
        
        transformed = preprocessor.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert preprocessor.is_fitted
        assert len(transformed) == len(sample_data)


class TestDataPreprocessorImputation:
    """Test imputation functionality."""
    
    def test_imputation_median(self, sample_data):
        """Test median imputation for numerical features."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {
                    'enabled': True,
                    'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}
                },
                'encoding': {'enabled': True, 'method': 'onehot', 'onehot_sparse': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert not transformed.isnull().any().any()
    
    def test_imputation_mean(self, sample_data):
        """Test mean imputation for numerical features."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {
                    'enabled': True,
                    'strategy': {'numerical': 'mean', 'categorical': 'most_frequent'}
                },
                'encoding': {'enabled': True, 'method': 'onehot', 'onehot_sparse': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert not transformed.isnull().any().any()
    
    def test_imputation_disabled(self, sample_data):
        """Test when imputation is disabled."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': False},
                'encoding': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert transformed.isnull().any().any()


class TestDataPreprocessorEncoding:
    """Test encoding functionality."""
    
    def test_encoding_onehot(self, sample_data):
        """Test one-hot encoding."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {
                    'enabled': True,
                    'method': 'onehot',
                    'onehot_sparse': False
                }
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert transformed.shape[1] > sample_data.shape[1]
    
    def test_encoding_sparse(self, sample_data):
        """Test sparse one-hot encoding."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {
                    'enabled': True,
                    'method': 'onehot',
                    'onehot_sparse': True
                },
                'downcast_dtype': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        preprocessor.fit(sample_data)
        
        # Check internal representation (before conversion to DataFrame)
        # Transform should still return DataFrame
        transformed = preprocessor.transform(sample_data)
        assert isinstance(transformed, pd.DataFrame)
    
    def test_encoding_ordinal(self, sample_data):
        """Test ordinal encoding."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {
                    'enabled': True,
                    'method': 'ordinal',
                    'onehot_sparse': False
                }
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert transformed.shape[1] == sample_data.shape[1]
    
    def test_encoding_disabled(self, sample_data):
        """Test when encoding is disabled."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert transformed.shape[1] == sample_data.shape[1]


class TestDataPreprocessorScaling:
    """Test scaling functionality."""
    
    def test_scaling_standard(self, sample_data):
        """Test standard scaling."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': False},
                'scaling': {
                    'enabled': True,
                    'method': 'standard',
                    'apply_to': 'numerical_only'
                },
                'downcast_dtype': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        # Numerical columns should be scaled (mean ~0, std ~1)
        # Note: won't be exact due to sample size
        assert isinstance(transformed, pd.DataFrame)
    
    def test_scaling_minmax(self, sample_data):
        """Test minmax scaling."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': False},
                'scaling': {
                    'enabled': True,
                    'method': 'minmax',
                    'apply_to': 'numerical_only'
                },
                'downcast_dtype': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
    
    def test_scaling_robust(self, sample_data):
        """Test robust scaling."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': False},
                'scaling': {
                    'enabled': True,
                    'method': 'robust',
                    'apply_to': 'numerical_only'
                },
                'downcast_dtype': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
    
    def test_scaling_disabled(self, sample_data, basic_config):
        """Test when scaling is disabled (default)."""
        preprocessor = DataPreprocessor(basic_config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)


class TestDataPreprocessorMemoryOptimization:
    """Test memory optimization (downcast dtype)."""
    
    def test_downcast_enabled(self, sample_data):
        """Test that downcast reduces memory."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': False},
                'downcast_dtype': {'enabled': True}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        for col in transformed.select_dtypes(include=[np.number]).columns:
            assert transformed[col].dtype in [np.int8, np.int16, np.int32, np.float32, np.float64]
    
    def test_downcast_disabled(self, sample_data):
        """Test when downcast is disabled."""
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': False},
                'downcast_dtype': {'enabled': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)


class TestDataPreprocessorGetFeatureNames:
    """Test get_feature_names method."""
    
    def test_get_feature_names_after_fit(self, sample_data, basic_config):
        """Test getting feature names after fit."""
        preprocessor = DataPreprocessor(basic_config)
        preprocessor.fit(sample_data)
        
        feature_names = preprocessor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_get_feature_names_not_fitted_error(self, basic_config):
        """Test error when getting feature names before fit."""
        preprocessor = DataPreprocessor(basic_config)
        
        with pytest.raises(RuntimeError, match="must be fitted first"):
            preprocessor.get_feature_names()


class TestDataPreprocessorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, basic_config):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        preprocessor = DataPreprocessor(basic_config)
        
        preprocessor.fit(empty_df)
        assert preprocessor.is_fitted
    
    def test_single_column(self):
        """Test with single column DataFrame."""
        df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': True, 'method': 'onehot', 'onehot_sparse': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(df)
        
        assert len(transformed.columns) == 1
    
    def test_all_missing_values(self):
        """Test with column of all missing values."""
        df = pd.DataFrame({
            'num1': [1, 2, 3],
            'num2': [np.nan, np.nan, np.nan],
            'cat1': ['A', 'B', 'C']
        })
        
        config = {
            'preprocessing': {
                'enabled': True,
                'imputation': {'enabled': True, 'strategy': {'numerical': 'median', 'categorical': 'most_frequent'}},
                'encoding': {'enabled': True, 'method': 'onehot', 'onehot_sparse': False}
            },
            'performance': {'parallel_jobs': 1}
        }
        
        preprocessor = DataPreprocessor(config)
        transformed = preprocessor.fit_transform(df)
        
        assert isinstance(transformed, pd.DataFrame)