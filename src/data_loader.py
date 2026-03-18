import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import logging
from pandas.errors import EmptyDataError

logger = logging.getLogger(__name__)

def load_data(
    file_path: str,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    dtype: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load data from CSV or Parquet with auto-detection.
    
    Args:
        file_path: Path to data file
        columns: Specific columns to load (None = all)
        nrows: Number of rows to load (None = all)
        dtype: Dictionary of column dtypes
    
    Returns:
        Loaded DataFrame
    
    Raises:
        FileNotFoundError: If file not found
        ValueError: If file format not supported
    """
    file = Path(file_path)
    
    if not file.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    file_ext = file.suffix.lower()
    
    logger.info(f"Loading data from {file_path}...")
    
    if file_ext == '.parquet':
        df = _load_parquet(file_path, columns)
    elif file_ext == '.csv':
        df = _load_csv(file_path, columns, nrows, dtype)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Use .csv or .parquet")
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def _load_parquet(
    file_path: str,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load Parquet file with optimizations.
    
    Args:
        file_path: Path to Parquet file
        columns: Columns to load (predicate pushdown)
    
    Returns:
        DataFrame
    """
    try:
        import pyarrow.parquet as pq
        
        df = pq.read_table(
            file_path,
            columns=columns,
            use_threads=True
        ).to_pandas()
        
        logger.debug("Loaded using pyarrow engine")
    
    except ImportError:
        df = pd.read_parquet(
            file_path,
            columns=columns,
            engine='auto'
        )
        
        logger.debug("Loaded using pandas engine")
    
    return df

def _load_csv(
    file_path: str,
    columns: Optional[list[str]] = None,
    nrows: Optional[int] = None,
    dtype: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load CSV file with optimizations.
    
    Args:
        file_path: Path to CSV file
        columns: Columns to load
        nrows: Number of rows to load
        dtype: Column data types
    
    Returns:
        DataFrame
    """
    try:
        df = pd.read_csv(
            file_path,
            usecols=columns,
            nrows=nrows,
            dtype=dtype,
            low_memory=False
        )
    except EmptyDataError:
        df = pd.DataFrame()
    
    return df

def load_train_test(
    train_path:str,
    test_path:str,
    target_column: str,
    id_column: Optional[str] = None,
    drop_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.Series]]:
    """
    Load train and test data with preprocessing.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        target_column: Name of target column
        id_column: Name of ID column (optional)
        drop_columns: Columns to drop
    
    Returns:
        Tuple of (X_train, y_train, X_test, test_ids)
    """
    train_df = load_data(train_path)
    logger.info(f"Train shape: {train_df.shape}")
    test_df = load_data(test_path)
    logger.info(f"Test shape: {test_df.shape}")
    
    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in training data")
    
    y_train = train_df[target_column].copy()
    
    test_ids = None
    if id_column and id_column in test_df.columns:
        test_ids = test_df[id_column].copy()
        
    cols_to_drop = [target_column]
    
    if id_column:
        cols_to_drop.append(id_column)
    
    if drop_columns:
        cols_to_drop.extend(drop_columns)
        
    cols_to_drop = list(set(cols_to_drop))
    
    X_train = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns])
    
    test_drop = [col for col in cols_to_drop if col in test_df.columns and col != target_column]
    X_test = test_df.drop(columns=test_drop)
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    
    return X_train, y_train, X_test, test_ids

def save_data(
    df: pd.DataFrame,
    file_path: str,
    format: str = 'parquet',
    compression: str = 'snappy'
) -> None:
    """
    Save DataFrame to file.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
        format: Output format ('parquet' or 'csv')
        compression: Compression method for Parquet
    """
    output_file = Path(file_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(
            output_file,
            compression=compression,
            index=False
        )
    elif format == 'csv':
        df.to_csv(
            output_file,
            index=False
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'parquet' or 'csv'.")
    
    logger.info(f"Data saved to {file_path} in {format} format.")