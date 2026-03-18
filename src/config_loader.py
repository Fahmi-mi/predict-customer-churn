import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

def load_config(config_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML config file with optional override.
    
    Args:
        config_path: Path to main config file
        override_path: Path to override config file (e.g., local.yaml)
    
    Returns:
        Merged configuration dictionary
    
    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    logger.info(f"Loaded config from {config_path}")
    
    if override_path:
        override_file = Path(override_path)
        if override_file.exists():
            with open(override_file, 'r', encoding='utf-8') as f:
                override_config = yaml.safe_load(f)
                
            config = deep_merge(config, override_config)
            logger.info(f"Merged override config from {override_path}")
            
    config = resolve_placeholders(config)
    
    validate_config(config)
    
    return config

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries (override values take precedence).
    
    Args:
        base: Base dictionary
        override: Override dictionary
    
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def resolve_placeholders(config: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Resolve placeholders like {experiment.name} in config values.
    
    Args:
        config: Configuration dictionary
        context: Context for placeholder resolution
    
    Returns:
        Config with resolved placeholders
    """
    if context is None:
        context = config.copy()
        
    def _resolve_value(value: Any) -> Any:
        if isinstance(value, str):
            pattern = r'\{([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\}'
            
            def replace_placeholder(match):
                keys = match.group(1).split('.')
                result = context
                
                try:
                    for key in keys:
                        result = result[key]
                    return str(result)
                except (KeyError, TypeError):
                    logger.warning(f"Placeholder not found: {match.group(0)}")
                    return match.group(0)
                
            return re.sub(pattern, replace_placeholder, value)
        elif isinstance(value, dict):
            return {k: _resolve_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_resolve_value(item) for item in value]
        else:
            return value
        
    return _resolve_value(config)

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required config fields.
    
    Args:
        config: Configuration dictionary
    
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_sections = ['experiment', 'data', 'preprocessing', 'output']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
        
    if 'name' not in config['experiment']:
        raise ValueError("Missing required field: experiment.name")

    if 'seed' not in config['experiment']:
        raise ValueError("Missing required field: experiment.seed")
    
    data_fields = ['train_path', 'target_column']
    for field in data_fields:
        if field not in config['data']:
            raise ValueError(f"Missing required field: data.{field}")
        
    if config.get('logging', {}).get('enabled', False):
        log_config = config['logging']
        if 'level' not in log_config:
            raise ValueError("Missing required field: logging.level")
        
    logger.info("Configuration validation passed.")
    
def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value by dot notation (e.g., 'preprocessing.imputation.enabled').
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path
        default: Default value if key not found
    
    Returns:
        Config value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
    
def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save config to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Config saved to: {output_path}")