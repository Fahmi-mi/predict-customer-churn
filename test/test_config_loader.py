import pytest
import yaml
from pathlib import Path
import tempfile
import shutil
import logging

from src.config_loader import (
    load_config,
    deep_merge,
    resolve_placeholders,
    validate_config,
    get_config_value,
    save_config
)


@pytest.fixture
def temp_config_dir():
    """Create temporary directory for config files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
    
@pytest.fixture
def sample_config():
    """Sample valid configuration."""
    return {
        'experiment': {
            'name': 'test_experiment',
            'seed': 42
        },
        'data': {
            'train_path': 'data/train.parquet',
            'test_path': 'data/test.parquet',
            'target_column': 'target',
            'id_column': 'id'
        },
        'preprocessing': {
            'enabled': True,
            'imputation': {
                'enabled': True,
                'strategy': {
                    'numerical': 'median',
                    'categorical': 'most_frequent'
                }
            }
        },
        'output': {
            'model_path': 'experiments/{experiment.name}/model.pkl',
            'submission_path': 'experiments/{experiment.name}/submission.csv'
        }
    }
    
@pytest.fixture
def sample_config_file(temp_config_dir, sample_config):
    """Create sample config YAML file."""
    config_path = Path(temp_config_dir) / "test_config.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
        
    return str(config_path)

class TestLoadConfig:
    """Test load_config function."""
    
    def test_load_config_basic(self, sample_config_file):
        """Test loading a valid config file."""
        config = load_config(sample_config_file)
        
        assert config['experiment']['name'] == 'test_experiment'
        assert config['experiment']['seed'] == 42
        assert config['data']['train_path'] == 'data/train.parquet'
        
    def test_load_config_file_not_found(self):
        """Test error when config file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config("nonexistent_config.yaml")
            
    def test_load_config_with_override(self, temp_config_dir, sample_config):
        """Test loading config with override."""
        base_path = Path(temp_config_dir) / "base.yaml"
        with open(base_path, 'w') as f:
            yaml.dump(sample_config, f)
            
        override_config = {
            'experiment': {
                'seed': 999
            },
            'data': {
                'train_path': 'data/override_train.parquet'
            }
        }
        override_path = Path(temp_config_dir) / "override.yaml"
        with open(override_path, 'w') as f:
            yaml.dump(override_config, f)
            
        config = load_config(str(base_path), str(override_path))
        
        assert config['experiment']['seed'] == 999
        assert config['data']['train_path'] == 'data/override_train.parquet'
        assert config['experiment']['name'] == 'test_experiment'
        
    def test_load_config_placeholder_resolution(self, temp_config_dir):
        """Test that placeholders are resolved."""
        config_data = {
            'experiment': {
                'name': 'my_experiment',
                'seed': 42
            },
            'data': {
                'train_path': 'data/train.parquet',
                'target_column': 'target'
            },
            'preprocessing': {'enabled': True},
            'output': {
                'model_path': 'experiments/{experiment.name}/model.pkl'
            }
        }
        
        config_path = Path(temp_config_dir) / "placeholder_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(str(config_path))
        
        # Placeholder should be resolved
        assert config['output']['model_path'] == 'experiments/my_experiment/model.pkl'
        
    def test_load_config_invalid_yaml(self, temp_config_dir):
        """Test error with invalid YAML."""
        invalid_path = Path(temp_config_dir) / "invalid.yaml"
        with open(invalid_path, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        with pytest.raises(yaml.YAMLError):
            load_config(str(invalid_path))


class TestDeepMerge:
    """Test deep_merge function."""
    
    def test_deep_merge_simple(self):
        """Test merging simple dictionaries."""
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        
        result = deep_merge(base, override)
        
        assert result == {'a': 1, 'b': 3, 'c': 4}
    
    def test_deep_merge_nested(self):
        """Test merging nested dictionaries."""
        base = {
            'level1': {
                'level2': {
                    'a': 1,
                    'b': 2
                },
                'c': 3
            }
        }
        override = {
            'level1': {
                'level2': {
                    'b': 999
                },
                'd': 4
            }
        }
        
        result = deep_merge(base, override)
        
        assert result['level1']['level2']['a'] == 1
        assert result['level1']['level2']['b'] == 999
        assert result['level1']['c'] == 3
        assert result['level1']['d'] == 4
    
    def test_deep_merge_different_types(self):
        """Test that override replaces when types differ."""
        base = {'a': {'nested': 1}}
        override = {'a': 'string_value'}
        
        result = deep_merge(base, override)
        
        assert result['a'] == 'string_value'
    
    def test_deep_merge_empty_dicts(self):
        """Test merging with empty dictionaries."""
        base = {'a': 1}
        override = {}
        
        result = deep_merge(base, override)
        assert result == {'a': 1}
        
        base = {}
        override = {'b': 2}
        
        result = deep_merge(base, override)
        assert result == {'b': 2}


class TestResolvePlaceholders:
    """Test resolve_placeholders function."""
    
    def test_resolve_placeholders_simple(self):
        """Test simple placeholder resolution."""
        config = {
            'experiment': {'name': 'test_exp'},
            'output': {'path': '{experiment.name}/results'}
        }
        
        result = resolve_placeholders(config)
        
        assert result['output']['path'] == 'test_exp/results'
    
    def test_resolve_placeholders_nested(self):
        """Test nested placeholder resolution."""
        config = {
            'experiment': {
                'name': 'exp1',
                'version': 'v1'
            },
            'paths': {
                'model': 'models/{experiment.name}/{experiment.version}/model.pkl',
                'logs': 'logs/{experiment.name}/run.log'
            }
        }
        
        result = resolve_placeholders(config)
        
        assert result['paths']['model'] == 'models/exp1/v1/model.pkl'
        assert result['paths']['logs'] == 'logs/exp1/run.log'
    
    def test_resolve_placeholders_multiple_in_string(self):
        """Test multiple placeholders in one string."""
        config = {
            'experiment': {'name': 'exp1', 'seed': 42},
            'output': 'experiments/{experiment.name}_seed_{experiment.seed}'
        }
        
        result = resolve_placeholders(config)
        
        assert result['output'] == 'experiments/exp1_seed_42'
    
    def test_resolve_placeholders_in_list(self):
        """Test placeholder resolution in lists."""
        config = {
            'experiment': {'name': 'exp1'},
            'files': [
                '{experiment.name}/file1.txt',
                '{experiment.name}/file2.txt'
            ]
        }
        
        result = resolve_placeholders(config)
        
        assert result['files'][0] == 'exp1/file1.txt'
        assert result['files'][1] == 'exp1/file2.txt'
    
    def test_resolve_placeholders_not_found(self):
        """Test that missing placeholders are left unchanged."""
        config = {
            'experiment': {'name': 'exp1'},
            'output': '{nonexistent.key}/results'
        }
        
        result = resolve_placeholders(config)
        
        assert result['output'] == '{nonexistent.key}/results'
    
    def test_resolve_placeholders_no_placeholders(self):
        """Test config without placeholders."""
        config = {
            'experiment': {'name': 'exp1'},
            'output': 'static/path/results'
        }
        
        result = resolve_placeholders(config)
        
        assert result['output'] == 'static/path/results'


class TestValidateConfig:
    """Test validate_config function."""
    
    def test_validate_config_valid(self, sample_config):
        """Test validation with valid config."""
        # Should not raise any exception
        validate_config(sample_config)
    
    def test_validate_config_missing_section(self):
        """Test validation with missing required section."""
        invalid_config = {
            'experiment': {'name': 'test', 'seed': 42},
            'preprocessing': {'enabled': True},
            'output': {}
        }
        
        with pytest.raises(ValueError, match="Missing required config section: data"):
            validate_config(invalid_config)
    
    def test_validate_config_missing_experiment_name(self):
        """Test validation with missing experiment.name."""
        invalid_config = {
            'experiment': {'seed': 42},
            'data': {'train_path': 'path', 'target_column': 'target'},
            'preprocessing': {'enabled': True},
            'output': {}
        }
        
        with pytest.raises(ValueError, match="Missing required field: experiment.name"):
            validate_config(invalid_config)
    
    def test_validate_config_missing_experiment_seed(self):
        """Test validation with missing experiment.seed."""
        invalid_config = {
            'experiment': {'name': 'test'},
            'data': {'train_path': 'path', 'target_column': 'target'},
            'preprocessing': {'enabled': True},
            'output': {}
        }
        
        with pytest.raises(ValueError, match="Missing required field: experiment.seed"):
            validate_config(invalid_config)
    
    def test_validate_config_missing_data_fields(self):
        """Test validation with missing data fields."""
        invalid_config = {
            'experiment': {'name': 'test', 'seed': 42},
            'data': {'train_path': 'path'},
            'preprocessing': {'enabled': True},
            'output': {}
        }
        
        with pytest.raises(ValueError, match="Missing required field: data.target_column"):
            validate_config(invalid_config)
    
    def test_validate_config_logging_missing_level(self):
        """Test validation with logging enabled but missing level."""
        invalid_config = {
            'experiment': {'name': 'test', 'seed': 42},
            'data': {'train_path': 'path', 'target_column': 'target'},
            'preprocessing': {'enabled': True},
            'output': {},
            'logging': {'enabled': True}
        }
        
        with pytest.raises(ValueError, match="Missing required field: logging.level"):
            validate_config(invalid_config)


class TestGetConfigValue:
    """Test get_config_value function."""
    
    def test_get_config_value_simple(self):
        """Test getting simple value."""
        config = {'experiment': {'name': 'test_exp', 'seed': 42}}
        
        result = get_config_value(config, 'experiment.name')
        
        assert result == 'test_exp'
    
    def test_get_config_value_nested(self):
        """Test getting deeply nested value."""
        config = {
            'preprocessing': {
                'imputation': {
                    'strategy': {
                        'numerical': 'median'
                    }
                }
            }
        }
        
        result = get_config_value(config, 'preprocessing.imputation.strategy.numerical')
        
        assert result == 'median'
    
    def test_get_config_value_not_found(self):
        """Test getting non-existent value returns default."""
        config = {'experiment': {'name': 'test'}}
        
        result = get_config_value(config, 'nonexistent.key', default='default_value')
        
        assert result == 'default_value'
    
    def test_get_config_value_default_none(self):
        """Test default is None when not specified."""
        config = {'experiment': {'name': 'test'}}
        
        result = get_config_value(config, 'nonexistent.key')
        
        assert result is None
    
    def test_get_config_value_partial_path(self):
        """Test with partial path that doesn't exist."""
        config = {'experiment': {'name': 'test'}}
        
        result = get_config_value(config, 'experiment.nonexistent.nested', default='fallback')
        
        assert result == 'fallback'


class TestSaveConfig:
    """Test save_config function."""
    
    def test_save_config_basic(self, temp_config_dir, sample_config):
        """Test saving config to file."""
        output_path = Path(temp_config_dir) / "saved_config.yaml"
        
        save_config(sample_config, str(output_path))
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded['experiment']['name'] == sample_config['experiment']['name']
        assert loaded['experiment']['seed'] == sample_config['experiment']['seed']
    
    def test_save_config_creates_directory(self, temp_config_dir, sample_config):
        """Test that save_config creates directories if needed."""
        output_path = Path(temp_config_dir) / "nested" / "dir" / "config.yaml"
        
        save_config(sample_config, str(output_path))
        
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_save_config_preserves_structure(self, temp_config_dir):
        """Test that nested structure is preserved."""
        complex_config = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 'deep_value'
                    }
                }
            },
            'list_data': [1, 2, 3],
            'mixed': {
                'int': 42,
                'float': 3.14,
                'bool': True,
                'none': None
            }
        }
        
        output_path = Path(temp_config_dir) / "complex_config.yaml"
        save_config(complex_config, str(output_path))
        
        with open(output_path, 'r') as f:
            loaded = yaml.safe_load(f)
        
        assert loaded['level1']['level2']['level3']['value'] == 'deep_value'
        assert loaded['list_data'] == [1, 2, 3]
        assert loaded['mixed']['int'] == 42
        assert loaded['mixed']['bool'] is True
        assert loaded['mixed']['none'] is None