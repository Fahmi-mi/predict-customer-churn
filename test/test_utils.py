import pytest
import logging
import os
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from src.utils import (
    set_seed,
    setup_logger,
    reduce_mem_usage,
    clear_memory,
    log_system_info,
    ensure_dir,
    format_time
)


class TestSetSeed:
    """Test set_seed function for reproducibility."""
    
    def test_set_seed_reproducibility(self):
        """Test that same seed produces same random numbers."""
        set_seed(42)
        random_1 = np.random.rand(5)
        
        set_seed(42)
        random_2 = np.random.rand(5)
        
        np.testing.assert_array_equal(random_1, random_2)
        
    def test_set_seed_different_values(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        random_1 = np.random.rand(5)
        
        set_seed(123)
        random_2 = np.random.rand(5)
        
        assert not np.array_equal(random_1, random_2)
        
class TestSetupLogger:
    """Test setup_logger function."""
    
    @pytest.fixture
    def temp_log_dir(self):
        """Create temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        for logger_obj in list(logging.Logger.manager.loggerDict.values()):
            if isinstance(logger_obj, logging.Logger):
                for handler in logger_obj.handlers[:]:
                    handler.close()
                    logger_obj.removeHandler(handler)
        shutil.rmtree(temp_dir)
        
    def test_setup_logger_console_only(self):
        """Test logger with console handler only."""
        logger = setup_logger(
            name="test_console",
            log_to_file=False,
            log_to_console=True,
        )
        
        assert logger.name == "test_console"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        
    def test_setup_logger_file_only(self, temp_log_dir):
        """Test logger with file handler only."""
        logger = setup_logger(
            name="test_file",
            log_to_file=True,
            log_to_console=False,
            log_dir=temp_log_dir,
            log_filename="test.log"
        )
        
        assert logger.name == "test_file"
        assert len(logger.handlers) >= 1
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        
        log_file = Path(temp_log_dir) / "test.log"
        assert log_file.exists()
        
    def test_setup_logger_both_handlers(self, temp_log_dir):
        """Test logger with both console and file handlers."""
        logger = setup_logger(
            name="test_both",
            log_to_file=True,
            log_to_console=True,
            log_dir=temp_log_dir,
            log_filename="test_both.log"
        )
        
        assert len(logger.handlers) >= 2
        
    def test_setup_logger_log_levels(self):
        """Test different log levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in levels:
            logger = setup_logger(
                name=f"test_{level}",
                log_level=level,
                log_to_file=False,
            )
            
            assert logger.level == getattr(logging, level)
            
    def test_setup_logger_creates_directory(self, temp_log_dir):
        """Test that logger creates log directory if not exists."""
        nested_dir = Path(temp_log_dir) / "nested" / "logs"
        
        logger = setup_logger(
            name="test_nested",
            log_to_file=True,
            log_to_console=False,
            log_dir=str(nested_dir)
        )
        
        assert nested_dir.exists()
        
class TestReduceMemUsage:
    """Test reduce_mem_usage function."""
    
    def test_reduce_mem_usage_int_types(self):
        """Test downcast of integer types."""
        df = pd.DataFrame({
            'small_int': [1, 2, 3, 4, 5],
            'medium_int': [1000, 2000, 3000, 4000, 5000],
            'large_int': [100000, 200000, 300000, 400000, 500000]
        })
        
        df_reduced = reduce_mem_usage(df, verbose=False)
        
        assert df_reduced['small_int'].dtype == np.int8
        assert df_reduced['medium_int'].dtype == np.int16
        assert df_reduced['large_int'].dtype == np.int32
        
    def test_reduce_mem_usage_float_types(self):
        """Test downcast of float types."""
        df = pd.DataFrame({
            'float_col': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        
        df_reduced = reduce_mem_usage(df, verbose=False)
        
        assert df_reduced['float_col'].dtype == np.float32
        
    def test_reduce_mem_usage_mixed_types(self):
        """Test with mixed types including objects."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'string_col': ['a', 'b', 'c']
        })
        
        original_mem = df.memory_usage().sum()
        df_reduced = reduce_mem_usage(df, verbose=False)
        reduced_mem = df_reduced.memory_usage().sum()
        
        assert reduced_mem < original_mem
        assert df_reduced['string_col'].dtype == object
        
    def test_reduce_mem_usage_verbose(self, capsys):
        """Test verbose output."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5] * 1000
        })
        
        reduce_mem_usage(df, verbose=True)
        
        captured = capsys.readouterr()
        assert "Memory usage decreased" in captured.out
        assert "MB" in captured.out
        assert "%" in captured.out
        
class TestClearMemory:
    """Test clear_memory function."""
    
    @patch('gc.collect')
    def test_clear_memory_calls_gc(self, mock_gc):
        """Test that clear_memory calls garbage collector."""
        clear_memory()
        
        mock_gc.assert_called_once()
        
class TestLogSystemInfo:
    """Test log_system_info function."""
    
    def test_log_system_info_output(self, caplog):
        """Test that system info is logged correctly."""
        logger = logging.getLogger("test_sysinfo")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            log_system_info(logger)
            
        log_text = caplog.text
        assert "SYSTEM INFORMATION" in log_text
        assert "Python version" in log_text
        assert "Platform" in log_text
        assert "Processor" in log_text
        
class TestEnsureDir:
    """Test ensure_dir function."""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
        
    def test_ensure_dir_creates_directory(self, temp_base_dir):
        """Test that directory is created."""
        new_dir = Path(temp_base_dir) / "new_directory"
        
        result = ensure_dir(str(new_dir))
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
        
    def test_ensure_dir_nested_directories(self, temp_base_dir):
        """Test that nested directories are created."""
        nested_dir = Path(temp_base_dir) / "level1" / "level2" / "level3"
        
        result = ensure_dir(str(nested_dir))
        
        assert nested_dir.exists()
        assert result == nested_dir
        
    def test_ensure_dir_existing_directory(self, temp_base_dir):
        """Test that existing directory doesn't raise error."""
        existing_dir = Path(temp_base_dir)
        
        result = ensure_dir(str(existing_dir))
        
        assert result == existing_dir
        
class TestFormatTime:
    """Test format_time function."""
    
    def test_format_time_seconds_only(self):
        """Test formatting for less than 1 minute."""
        assert format_time(30) == "30s"
        assert format_time(59) == "59s"
    
    def test_format_time_minutes_seconds(self):
        """Test formatting for minutes and seconds."""
        assert format_time(90) == "1m 30s"
        assert format_time(125) == "2m 5s"
        assert format_time(3599) == "59m 59s"
    
    def test_format_time_hours_minutes_seconds(self):
        """Test formatting for hours, minutes, and seconds."""
        assert format_time(3661) == "1h 1m 1s"
        assert format_time(7200) == "2h 0m 0s"
        assert format_time(7325) == "2h 2m 5s"
    
    def test_format_time_zero(self):
        """Test formatting for zero seconds."""
        assert format_time(0) == "0s"
    
    def test_format_time_float_input(self):
        """Test with float input."""
        assert format_time(90.7) == "1m 30s"
        assert format_time(3661.9) == "1h 1m 1s"