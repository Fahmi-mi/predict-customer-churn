import pytest
import pandas as pd
import numpy as np
import logging

from src.feature_engineering import FeatureEngineer, engineer_features


@pytest.fixture
def sample_data():
    """Create sample dataset for feature engineering."""
    np.random.seed(42)
    
    data = {
        'Age': [25, 30, 35, 40, 45, 50, 55, 60],
        'Fare': [10.5, 20.3, 15.7, 40.1, 50.8, 60.2, 35.5, 80.5],
        'Pclass': [1, 2, 3, 1, 2, 3, 1, 2],
        'SibSp': [0, 1, 0, 2, 1, 0, 3, 1],
        'Parch': [0, 0, 1, 1, 2, 0, 1, 0]
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def basic_config():
    """Basic configuration for feature engineering."""
    return {
        'feature_engineering': {
            'enabled': True,
            'interactions': [],
            'binning': {}
        }
    }


class TestFeatureEngineerInit:
    """Test FeatureEngineer initialization."""
    
    def test_init_basic(self, basic_config):
        """Test basic initialization."""
        engineer = FeatureEngineer(basic_config)
        
        assert engineer.config == basic_config
        assert engineer.fe_config == basic_config['feature_engineering']
        assert engineer.interactions == []
        assert engineer.binning == {}
        assert engineer.created_features == []
    
    def test_init_with_interactions(self):
        """Test initialization with interactions specified."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare', 'Pclass*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        
        assert engineer.interactions == ['Age*Fare', 'Pclass*Fare']
    
    def test_init_with_binning(self):
        """Test initialization with binning specified."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 18, 35, 60, 100],
                    'Fare': [0, 10, 30, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        
        assert 'Age' in engineer.binning
        assert 'Fare' in engineer.binning


class TestFeatureEngineerFit:
    """Test FeatureEngineer fit method."""
    
    def test_fit_basic(self, sample_data, basic_config):
        """Test basic fit operation."""
        engineer = FeatureEngineer(basic_config)
        
        result = engineer.fit(sample_data)
        
        assert result is engineer
    
    def test_fit_validates_interactions(self, sample_data):
        """Test that fit validates interaction specifications."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        
        engineer.fit(sample_data)
    
    def test_fit_invalid_interaction_format(self, sample_data):
        """Test that invalid interaction format raises error."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age+Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        
        with pytest.raises(ValueError, match="Invalid interaction format"):
            engineer.fit(sample_data)
    
    def test_fit_validates_binning(self, sample_data):
        """Test that fit validates binning specifications."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 18, 35, 60, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        
        engineer.fit(sample_data)
    
    def test_fit_invalid_binning_format(self, sample_data):
        """Test that invalid binning format raises error."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        
        with pytest.raises(ValueError, match="Invalid bins"):
            engineer.fit(sample_data)


class TestFeatureEngineerTransform:
    """Test FeatureEngineer transform method."""
    
    def test_transform_basic(self, sample_data, basic_config):
        """Test basic transform returns copy of data when no operations."""
        engineer = FeatureEngineer(basic_config)
        engineer.fit(sample_data)
        
        transformed = engineer.transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)
        pd.testing.assert_frame_equal(transformed, sample_data)
    
    def test_transform_disabled(self, sample_data):
        """Test transform when feature engineering is disabled."""
        config = {
            'feature_engineering': {
                'enabled': False,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        engineer.fit(sample_data)
        
        transformed = engineer.transform(sample_data)
        
        pd.testing.assert_frame_equal(transformed, sample_data)
        assert len(engineer.created_features) == 0


class TestFeatureEngineerFitTransform:
    """Test FeatureEngineer fit_transform method."""
    
    def test_fit_transform_basic(self, sample_data, basic_config):
        """Test fit_transform combines fit and transform."""
        engineer = FeatureEngineer(basic_config)
        
        transformed = engineer.fit_transform(sample_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_data)


class TestFeatureEngineerInteractions:
    """Test interaction feature creation."""
    
    def test_create_single_interaction(self, sample_data):
        """Test creating a single interaction feature."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age_x_Fare' in transformed.columns
        assert len(transformed.columns) == len(sample_data.columns) + 1
        
        expected = sample_data['Age'] * sample_data['Fare']
        pd.testing.assert_series_equal(
            transformed['Age_x_Fare'],
            expected,
            check_names=False
        )
    
    def test_create_multiple_interactions(self, sample_data):
        """Test creating multiple interaction features."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare', 'Pclass*Fare', 'SibSp*Parch'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age_x_Fare' in transformed.columns
        assert 'Pclass_x_Fare' in transformed.columns
        assert 'SibSp_x_Parch' in transformed.columns
        assert len(transformed.columns) == len(sample_data.columns) + 3
    
    def test_interaction_with_whitespace(self, sample_data):
        """Test interaction with whitespace in specification."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age * Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age_x_Fare' in transformed.columns
    
    def test_interaction_feature_not_found(self, sample_data, caplog):
        """Test interaction with non-existent feature logs warning."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['NonExistent*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        
        with caplog.at_level(logging.WARNING):
            transformed = engineer.fit_transform(sample_data)
        
        # Should skip invalid interaction
        assert 'NonExistent_x_Fare' not in transformed.columns
        assert "Skipping interaction" in caplog.text
    
    def test_interaction_preserves_original_features(self, sample_data):
        """Test that interactions preserve original features."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        for col in sample_data.columns:
            assert col in transformed.columns


class TestFeatureEngineerBinning:
    """Test binning feature creation."""
    
    def test_create_single_binning(self, sample_data):
        """Test creating a single binned feature."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 18, 35, 60, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age_binned' in transformed.columns
        assert len(transformed.columns) == len(sample_data.columns) + 1
        
        assert transformed['Age_binned'].dtype == object
    
    def test_create_multiple_binnings(self, sample_data):
        """Test creating multiple binned features."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 18, 35, 60, 100],
                    'Fare': [0, 20, 50, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age_binned' in transformed.columns
        assert 'Fare_binned' in transformed.columns
        assert len(transformed.columns) == len(sample_data.columns) + 2
    
    def test_binning_with_edge_values(self, sample_data):
        """Test binning handles edge values correctly."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 30, 50, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert not transformed['Age_binned'].isnull().any()
    
    def test_binning_feature_not_found(self, sample_data, caplog):
        """Test binning with non-existent feature logs warning."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'NonExistent': [0, 10, 20]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        
        with caplog.at_level(logging.WARNING):
            transformed = engineer.fit_transform(sample_data)
        
        assert 'NonExistent_binned' not in transformed.columns
        assert "Skipping binning" in caplog.text
    
    def test_binning_preserves_original_features(self, sample_data):
        """Test that binning preserves original features."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 18, 35, 60, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age' in transformed.columns
        pd.testing.assert_series_equal(transformed['Age'], sample_data['Age'])


class TestFeatureEngineerCombined:
    """Test combined interactions and binning."""
    
    def test_interactions_and_binning(self, sample_data):
        """Test creating both interactions and binning."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare', 'Pclass*Fare'],
                'binning': {
                    'Age': [0, 18, 35, 60, 100],
                    'Fare': [0, 20, 50, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(sample_data)
        
        assert 'Age_x_Fare' in transformed.columns
        assert 'Pclass_x_Fare' in transformed.columns
        assert 'Age_binned' in transformed.columns
        assert 'Fare_binned' in transformed.columns
        
        assert len(transformed.columns) == len(sample_data.columns) + 4
        assert len(engineer.created_features) == 4


class TestFeatureEngineerGetCreatedFeatures:
    """Test get_created_features method."""
    
    def test_get_created_features_empty(self, sample_data, basic_config):
        """Test getting created features when none created."""
        engineer = FeatureEngineer(basic_config)
        engineer.fit_transform(sample_data)
        
        created = engineer.get_created_features()
        
        assert created == []
    
    def test_get_created_features_with_interactions(self, sample_data):
        """Test getting created features with interactions."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        engineer.fit_transform(sample_data)
        
        created = engineer.get_created_features()
        
        assert 'Age_x_Fare' in created
        assert len(created) == 1
    
    def test_get_created_features_with_binning(self, sample_data):
        """Test getting created features with binning."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': [],
                'binning': {
                    'Age': [0, 18, 35, 60, 100]
                }
            }
        }
        
        engineer = FeatureEngineer(config)
        engineer.fit_transform(sample_data)
        
        created = engineer.get_created_features()
        
        assert 'Age_binned' in created
        assert len(created) == 1


class TestEngineerFeaturesFunction:
    """Test engineer_features convenience function."""
    
    def test_engineer_features_basic(self, sample_data):
        """Test engineer_features function with train and test data."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {
                    'Age': [0, 18, 35, 60, 100]
                }
            }
        }
        
        X_train = sample_data.iloc[:6]
        X_test = sample_data.iloc[6:]
        
        X_train_fe, X_test_fe, engineer = engineer_features(X_train, X_test, config)
        
        # Check train data
        assert isinstance(X_train_fe, pd.DataFrame)
        assert 'Age_x_Fare' in X_train_fe.columns
        assert 'Age_binned' in X_train_fe.columns
        
        # Check test data
        assert isinstance(X_test_fe, pd.DataFrame)
        assert 'Age_x_Fare' in X_test_fe.columns
        assert 'Age_binned' in X_test_fe.columns
        
        # Check engineer
        assert isinstance(engineer, FeatureEngineer)
        assert len(engineer.get_created_features()) == 2
    
    def test_engineer_features_consistency(self, sample_data):
        """Test that train and test get same transformations."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        X_train = sample_data.iloc[:6]
        X_test = sample_data.iloc[6:]
        
        X_train_fe, X_test_fe, _ = engineer_features(X_train, X_test, config)
        
        assert set(X_train_fe.columns) == set(X_test_fe.columns)


class TestFeatureEngineerEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, basic_config):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        engineer = FeatureEngineer(basic_config)
        transformed = engineer.fit_transform(empty_df)
        
        assert len(transformed) == 0
    
    def test_single_row(self):
        """Test with single row DataFrame."""
        df = pd.DataFrame({'Age': [25], 'Fare': [10.5]})
        
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        transformed = engineer.fit_transform(df)
        
        assert 'Age_x_Fare' in transformed.columns
        assert len(transformed) == 1
    
    def test_multiple_transforms(self, sample_data):
        """Test that transform can be called multiple times."""
        config = {
            'feature_engineering': {
                'enabled': True,
                'interactions': ['Age*Fare'],
                'binning': {}
            }
        }
        
        engineer = FeatureEngineer(config)
        engineer.fit(sample_data)
        
        transformed1 = engineer.transform(sample_data)
        transformed2 = engineer.transform(sample_data)
        
        pd.testing.assert_frame_equal(transformed1, transformed2)