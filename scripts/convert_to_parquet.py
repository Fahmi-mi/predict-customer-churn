import pandas as pd
from pathlib import Path
import logging
from typing import List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import setup_logger, format_time
import time

logger = setup_logger(
    name='convert_to_parquet',
    log_level='INFO',
    log_to_file=True,
    log_to_console=True,
    log_dir='logs',
    log_filename='convert_to_parquet.log'
)

def convert_csv_to_parquet(
    csv_path: Path,
    output_dir: Path,
    compression: str = 'snappy'
) -> None:
    """
    Convert single CSV file to Parquet.
    
    Args:
        csv_path: Path to CSV file
        output_dir: Output directory for Parquet file
        compression: Compression method ('snappy', 'gzip', 'brotli', 'none')
    """
    logger.info(f"Converting: {csv_path.name}")
    start_time = time.time()
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        csv_size_mb = csv_path.stat().st_size / (1024 * 1024)
        logger.info(f"  CSV size: {csv_size_mb:.2f} MB")
        
        output_path = output_dir / csv_path.with_suffix('.parquet').name
        df.to_parquet(
            output_path,
            compression=compression,
            index=False,
            engine='pyarrow'
        )
        
        parquet_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Parquet size: {parquet_size_mb:.2f} MB")
        
        compression_ratio = (1 - parquet_size_mb / csv_size_mb) * 100
        elapsed = time.time() - start_time
        
        logger.info(f"  Compression: {compression_ratio:.1f}% reduction")
        logger.info(f"  Time: {format_time(elapsed)}")
        logger.info(f"  Saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"  Error converting {csv_path.name}: {e}")
        raise
    
def find_csv_files(directory: Path) -> List[Path]:
    """
    Find all CSV files in directory.
    
    Args:
        directory: Directory to search
    
    Returns:
        List of CSV file paths
    """
    csv_files = list(directory.glob('*.csv'))
    return csv_files

def main():
    """
    Main conversion process.
    """
    logger.info("="*60)
    logger.info("CSV to Parquet Conversion")
    logger.info("="*60)
    
    raw_dir = Path('data/raw')
    processed_dir = Path('data/processed')
    
    if not raw_dir.exists():
        logger.error(f"Input directory not found: {raw_dir}")
        logger.info("Please create 'data/raw/' and place CSV files there")
        return
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    csv_files = find_csv_files(raw_dir)
    
    if not csv_files:
        logger.warning(f"No CSV files found in {raw_dir}")
        logger.info("Please place CSV files (e.g., train.csv, test.csv) in data/raw/")
        return
    
    logger.info(f"Found {len(csv_files)} CSV file(s) to convert")
    logger.info("")
    
    total_start = time.time()
    success_count = 0
    
    for csv_file in csv_files:
        try:
            convert_csv_to_parquet(csv_file, processed_dir)
            success_count += 1
            logger.info("")
        except Exception as e:
            logger.error(f"Failed to convert {csv_file.name}")
            logger.info("")
    
    total_elapsed = time.time() - total_start
    
    logger.info("="*60)
    logger.info(f"Conversion complete: {success_count}/{len(csv_files)} files converted")
    logger.info(f"Total time: {format_time(total_elapsed)}")
    logger.info(f"Output directory: {processed_dir.absolute()}")
    logger.info("="*60)


if __name__ == '__main__':
    main()