import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import logging
import importlib.util

from src.data_loader import (
    load_data,
    _load_parquet,
    _load_csv,
    load_train_test,
    save_data
)


@pytest.fixture
def temp_data_dir():
    """Create temporary directory for test data files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)
    
@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    
    data = {
        'id': range(1, 101),
        'num1': np.random.randn(100),
        'num2': np.random.randint(0, 100, 100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100),
        'cat2': np.random.choice(['X', 'Y', 'Z'], 100),
        'target': np.random.randint(0, 2, 100)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_csv_file(temp_data_dir, sample_dataframe):
    """Create sample CSV file."""
    csv_path = Path(temp_data_dir) / "sample_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def sample_parquet_file(temp_data_dir, sample_dataframe):
    """Create sample Parquet file."""
    parquet_path = Path(temp_data_dir) / "sample_data.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False, compression='snappy')
    return str(parquet_path)

@pytest.fixture
def train_test_files(temp_data_dir, sample_dataframe):
    """Create train and test CSV files."""
    train_df = sample_dataframe.iloc[:70]
    test_df = sample_dataframe.iloc[70:].drop(columns=['target'])
    
    train_path = Path(temp_data_dir) / "train.csv"
    test_path = Path(temp_data_dir) / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    return str(train_path), str(test_path)


class TestDataLoader:
    """Test load_data function."""
    
    def test_load_csv_basic(self, sample_csv_file, sample_dataframe):
        """Test loading CSV file."""
        df = load_data(sample_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
        
    def test_load_parquet_basic(self, sample_parquet_file, sample_dataframe):
        """Test loading Parquet file."""
        df = load_data(sample_parquet_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
        
    def test_load_data_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            load_data("nonexistent_file.csv")
    
    def test_load_data_unsupported_format(self, temp_data_dir):
        """Test error with unsupported file format."""
        unsupported_file = Path(temp_data_dir) / "data.xlsx"
        unsupported_file.touch()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_data(str(unsupported_file))
    
    def test_load_csv_with_columns(self, sample_csv_file):
        """Test loading specific columns from CSV."""
        columns = ['id', 'num1', 'target']
        df = load_data(sample_csv_file, columns=columns)
        
        assert list(df.columns) == columns
        assert len(df.columns) == 3
    
    def test_load_parquet_with_columns(self, sample_parquet_file):
        """Test loading specific columns from Parquet (predicate pushdown)."""
        columns = ['id', 'num1', 'target']
        df = load_data(sample_parquet_file, columns=columns)
        
        assert list(df.columns) == columns
        assert len(df.columns) == 3
    
    def test_load_csv_with_nrows(self, sample_csv_file):
        """Test loading limited rows from CSV."""
        nrows = 20
        df = load_data(sample_csv_file, nrows=nrows)
        
        assert len(df) == nrows
    
    def test_load_csv_with_dtype(self, sample_csv_file):
        """Test loading CSV with specific dtypes."""
        dtype = {'num2': np.int32, 'cat1': 'category'}
        df = load_data(sample_csv_file, dtype=dtype)
        
        assert df['num2'].dtype == np.int32
        assert df['cat1'].dtype == 'category'
    
    def test_load_data_preserves_data(self, sample_csv_file, sample_dataframe):
        """Test that loaded data matches original."""
        df = load_data(sample_csv_file)
        
        pd.testing.assert_frame_equal(df, sample_dataframe, check_dtype=False)


class TestLoadParquet:
    """Test _load_parquet function."""
    
    def test_load_parquet_all_columns(self, sample_parquet_file, sample_dataframe):
        """Test loading all columns from Parquet."""
        df = _load_parquet(sample_parquet_file)
        
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
    
    def test_load_parquet_specific_columns(self, sample_parquet_file):
        """Test loading specific columns from Parquet."""
        columns = ['id', 'num1']
        df = _load_parquet(sample_parquet_file, columns=columns)
        
        assert list(df.columns) == columns
    
    def test_load_parquet_with_pyarrow(self, sample_parquet_file):
        """Test that pyarrow is used if available."""
        try:
            import pyarrow
            
            df = _load_parquet(sample_parquet_file)
            assert isinstance(df, pd.DataFrame)
        except ImportError:
            pytest.skip("pyarrow not installed")
    
    def test_load_parquet_fallback_pandas(self, sample_parquet_file, monkeypatch):
        """Test fallback to pandas engine when pyarrow unavailable."""
        if importlib.util.find_spec('fastparquet') is None:
            pytest.skip("fastparquet not installed; pandas parquet fallback engine unavailable")

        original_import = __import__

        def mock_import(*args, **kwargs):
            if args[0] == 'pyarrow.parquet':
                raise ImportError("Mocked pyarrow unavailable")
            return original_import(*args, **kwargs)
        
        monkeypatch.setattr('builtins.__import__', mock_import)
        
        df = _load_parquet(sample_parquet_file)
        assert isinstance(df, pd.DataFrame)


class TestLoadCSV:
    """Test _load_csv function."""
    
    def test_load_csv_all_data(self, sample_csv_file, sample_dataframe):
        """Test loading all data from CSV."""
        df = _load_csv(sample_csv_file)
        
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)
    
    def test_load_csv_with_columns(self, sample_csv_file):
        """Test loading specific columns from CSV."""
        columns = ['id', 'num1', 'target']
        df = _load_csv(sample_csv_file, columns=columns)
        
        assert list(df.columns) == columns
    
    def test_load_csv_with_nrows(self, sample_csv_file):
        """Test loading limited rows from CSV."""
        nrows = 10
        df = _load_csv(sample_csv_file, nrows=nrows)
        
        assert len(df) == nrows
    
    def test_load_csv_with_dtype(self, sample_csv_file):
        """Test loading CSV with specific dtypes."""
        dtype = {'num2': np.int32}
        df = _load_csv(sample_csv_file, dtype=dtype)
        
        assert df['num2'].dtype == np.int32


class TestLoadTrainTest:
    """Test load_train_test function."""
    
    def test_load_train_test_basic(self, train_test_files):
        """Test basic train/test loading."""
        train_path, test_path = train_test_files
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=train_path,
            test_path=test_path,
            target_column='target',
            id_column='id'
        )
        
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(test_ids, pd.Series)
        
        assert len(X_train) == 70
        assert len(y_train) == 70
        assert len(X_test) == 30
        assert len(test_ids) == 30
    
    def test_load_train_test_no_id_column(self, train_test_files):
        """Test loading without ID column."""
        train_path, test_path = train_test_files
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=train_path,
            test_path=test_path,
            target_column='target',
            id_column=None
        )
        
        assert test_ids is None
    
    def test_load_train_test_drop_columns(self, train_test_files):
        """Test dropping additional columns."""
        train_path, test_path = train_test_files
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=train_path,
            test_path=test_path,
            target_column='target',
            id_column='id',
            drop_columns=['cat1']
        )
        
        assert 'cat1' not in X_train.columns
        assert 'cat1' not in X_test.columns
    
    def test_load_train_test_target_not_in_train(self, train_test_files):
        """Test error when target column not in training data."""
        train_path, test_path = train_test_files
        
        with pytest.raises(ValueError, match="Target column .* not found"):
            load_train_test(
                train_path=train_path,
                test_path=test_path,
                target_column='nonexistent_target',
                id_column='id'
            )
    
    def test_load_train_test_removes_target_from_features(self, train_test_files):
        """Test that target is removed from X_train."""
        train_path, test_path = train_test_files
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=train_path,
            test_path=test_path,
            target_column='target',
            id_column='id'
        )
        
        assert 'target' not in X_train.columns
        
        assert y_train.name == 'target'
    
    def test_load_train_test_removes_id_from_features(self, train_test_files):
        """Test that ID is removed from features."""
        train_path, test_path = train_test_files
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=train_path,
            test_path=test_path,
            target_column='target',
            id_column='id'
        )
        
        assert 'id' not in X_train.columns
        assert 'id' not in X_test.columns
    
    def test_load_train_test_preserves_dtypes(self, train_test_files):
        """Test that data types are preserved."""
        train_path, test_path = train_test_files
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=train_path,
            test_path=test_path,
            target_column='target',
            id_column='id'
        )
        
        assert 'num1' in X_train.columns
        assert 'num2' in X_train.columns
        
        assert 'cat1' in X_train.columns
        assert 'cat2' in X_train.columns


class TestSaveData:
    """Test save_data function."""
    
    def test_save_data_parquet(self, temp_data_dir, sample_dataframe):
        """Test saving DataFrame to Parquet."""
        output_path = Path(temp_data_dir) / "output.parquet"
        
        save_data(sample_dataframe, str(output_path), format='parquet')
        
        assert output_path.exists()
        
        df_loaded = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df_loaded, sample_dataframe)
    
    def test_save_data_csv(self, temp_data_dir, sample_dataframe):
        """Test saving DataFrame to CSV."""
        output_path = Path(temp_data_dir) / "output.csv"
        
        save_data(sample_dataframe, str(output_path), format='csv')
        
        assert output_path.exists()
        
        df_loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(df_loaded, sample_dataframe, check_dtype=False)
    
    def test_save_data_creates_directory(self, temp_data_dir, sample_dataframe):
        """Test that save_data creates nested directories."""
        output_path = Path(temp_data_dir) / "nested" / "dir" / "output.parquet"
        
        save_data(sample_dataframe, str(output_path), format='parquet')
        
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_save_data_parquet_compression(self, temp_data_dir, sample_dataframe):
        """Test Parquet with different compression."""
        output_path = Path(temp_data_dir) / "compressed.parquet"
        
        save_data(
            sample_dataframe,
            str(output_path),
            format='parquet',
            compression='gzip'
        )
        
        assert output_path.exists()
        
        df_loaded = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df_loaded, sample_dataframe)
    
    def test_save_data_unsupported_format(self, temp_data_dir, sample_dataframe):
        """Test error with unsupported save format."""
        output_path = Path(temp_data_dir) / "output.xlsx"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            save_data(sample_dataframe, str(output_path), format='excel')
    
    def test_save_data_no_index(self, temp_data_dir, sample_dataframe):
        """Test that index is not saved."""
        output_path_parquet = Path(temp_data_dir) / "no_index.parquet"
        output_path_csv = Path(temp_data_dir) / "no_index.csv"
        
        save_data(sample_dataframe, str(output_path_parquet), format='parquet')
        save_data(sample_dataframe, str(output_path_csv), format='csv')
        
        df_parquet = pd.read_parquet(output_path_parquet)
        df_csv = pd.read_csv(output_path_csv)
        
        assert list(df_parquet.columns) == list(sample_dataframe.columns)
        assert list(df_csv.columns) == list(sample_dataframe.columns)


class TestLoadDataIntegration:
    """Integration tests for data loading workflow."""
    
    def test_csv_parquet_roundtrip(self, temp_data_dir, sample_dataframe):
        """Test converting CSV to Parquet and back."""
        csv_path = Path(temp_data_dir) / "original.csv"
        parquet_path = Path(temp_data_dir) / "converted.parquet"
        
        sample_dataframe.to_csv(csv_path, index=False)
        
        df_csv = load_data(str(csv_path))
        save_data(df_csv, str(parquet_path), format='parquet')
        
        df_parquet = load_data(str(parquet_path))
        
        pd.testing.assert_frame_equal(df_parquet, sample_dataframe, check_dtype=False)
    
    def test_train_test_workflow(self, temp_data_dir, sample_dataframe):
        """Test complete train/test loading and saving workflow."""
        train_df = sample_dataframe.iloc[:70]
        test_df = sample_dataframe.iloc[70:].drop(columns=['target'])
        
        train_path = Path(temp_data_dir) / "train.parquet"
        test_path = Path(temp_data_dir) / "test.parquet"
        
        save_data(train_df, str(train_path), format='parquet')
        save_data(test_df, str(test_path), format='parquet')
        
        X_train, y_train, X_test, test_ids = load_train_test(
            train_path=str(train_path),
            test_path=str(test_path),
            target_column='target',
            id_column='id'
        )
        
        assert len(X_train) == 70
        assert len(X_test) == 30
        assert 'target' not in X_train.columns
        assert 'id' not in X_train.columns


class TestLoadDataEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_load_empty_csv(self, temp_data_dir):
        """Test loading empty CSV file."""
        empty_df = pd.DataFrame()
        csv_path = Path(temp_data_dir) / "empty.csv"
        empty_df.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        assert len(df) == 0
    
    def test_load_single_row(self, temp_data_dir):
        """Test loading file with single row."""
        single_row_df = pd.DataFrame({'col1': [1], 'col2': ['A']})
        csv_path = Path(temp_data_dir) / "single_row.csv"
        single_row_df.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        assert len(df) == 1
        pd.testing.assert_frame_equal(df, single_row_df)
    
    def test_load_single_column(self, temp_data_dir):
        """Test loading file with single column."""
        single_col_df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        csv_path = Path(temp_data_dir) / "single_col.csv"
        single_col_df.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        assert len(df.columns) == 1
        pd.testing.assert_frame_equal(df, single_col_df)
    
    def test_load_with_missing_values(self, temp_data_dir):
        """Test loading data with missing values."""
        df_with_na = pd.DataFrame({
            'col1': [1, 2, np.nan, 4],
            'col2': ['A', None, 'C', 'D'],
            'col3': [1.5, 2.5, 3.5, np.nan]
        })
        
        csv_path = Path(temp_data_dir) / "with_na.csv"
        df_with_na.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        assert df.isnull().any().any()
    
    def test_load_large_integers(self, temp_data_dir):
        """Test loading data with large integers."""
        df_large_int = pd.DataFrame({
            'large_int': [10**15, 10**16, 10**17]
        })
        
        csv_path = Path(temp_data_dir) / "large_int.csv"
        df_large_int.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        assert isinstance(df, pd.DataFrame)
    
    def test_load_special_characters(self, temp_data_dir):
        """Test loading data with special characters."""
        df_special = pd.DataFrame({
            'text': ['Hello, World!', 'Test\nNewline', 'Tab\there', '"Quoted"']
        })
        
        csv_path = Path(temp_data_dir) / "special_chars.csv"
        df_special.to_csv(csv_path, index=False)
        
        df = load_data(str(csv_path))
        
        pd.testing.assert_frame_equal(df, df_special)
    
    def test_load_different_dtypes(self, temp_data_dir):
        """Test loading data with various data types."""
        df_dtypes = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.5, 2.5, 3.5],
            'str_col': ['A', 'B', 'C'],
            'bool_col': [True, False, True]
        })
        
        parquet_path = Path(temp_data_dir) / "dtypes.parquet"
        df_dtypes.to_parquet(parquet_path, index=False)
        
        df = load_data(str(parquet_path))
        
        assert df['int_col'].dtype in [np.int32, np.int64]
        assert df['float_col'].dtype in [np.float32, np.float64]
        assert df['str_col'].dtype == object
        assert df['bool_col'].dtype == bool