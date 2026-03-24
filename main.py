import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import shutil

from src.config_loader import load_config, save_config
from src.data_loader import load_train_test, save_data
from src.preprocessor import preprocess_data
from src.feature_engineering import engineer_features
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator
from src.predictor import predict_and_submit
from src.utils import (
    setup_logger, set_seed, log_system_info,
    format_time, ensure_dir, clear_memory
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='ML Pipeline for Tabular Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --config default
  python main.py --config experiment1 --mode train
  python main.py --config default --override model.type=catboost
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='default',
        help='Config file name (without .yaml extension)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['full', 'train', 'predict'],
        help='Execution mode: full (train+predict), train only, or predict only'
    )
    
    parser.add_argument(
        '--override',
        type=str,
        nargs='*',
        help='Override config values (e.g., model.type=catboost data.train_path=data.csv)'
    )
    
    return parser.parse_args()

def apply_config_overrides(config: Dict[str, Any], overrides: Optional[list]) -> Dict[str, Any]:
    """
    Apply command line overrides to config.
    
    Args:
        config: Configuration dictionary
        overrides: List of override strings (e.g., ['model.type=catboost'])
    
    Returns:
        Updated configuration
    """
    if not overrides:
        return config
    
    for override in overrides:
        if '=' not in override:
            print(f"Warning: Invalid override format: {override}. Use key=value")
            continue
        
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        final_key = keys[-1]
        if value.lower() == 'true':
            current[final_key] = True
        elif value.lower() == 'false':
            current[final_key] = False
        elif value.lower() == 'null':
            current[final_key] = None
        elif value.isdigit():
            current[final_key] = int(value)
        else:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value
                
        print(f"  Override: {key_path} = {current[final_key]}")
    
    return config

def setup_experiment_dir(config: Dict[str, Any]) -> Path:
    """
    Create experiment directory and save config snapshot.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Experiment directory path
    """
    exp_name = config['experiment']['name']
    exp_dir = Path(f"experiments/{exp_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_snapshot_path = exp_dir / 'config_snapshot.yaml'
    save_config(config, str(config_snapshot_path))
    
    return exp_dir

def run_pipeline(config: Dict[str, Any], mode: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Execute the ML pipeline.
    
    Args:
        config: Configuration dictionary
        mode: Execution mode ('full', 'train', 'predict')
        logger: Logger instance
    
    Returns:
        Dictionary with pipeline results and metrics
    """
    results = {}
    timings = {}
    
    exp_dir = setup_experiment_dir(config)
    logger.info(f"Experiment directory: {exp_dir}")
    
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'lightgbm')
    task_type = model_config.get('task_type', 'classification')
    
    # ============================================================
    # STAGE 1: LOAD DATA
    # ============================================================
    if mode in ['full', 'train']:
        logger.info("\n" + "="*60)
        logger.info("STAGE 1: LOADING DATA")
        logger.info("="*40)
        
        stage_start = time.time()
        
        try:
            X_train, y_train, X_test, test_ids = load_train_test(
                train_path=config['data']['train_path'],
                test_path=config['data']['test_path'],
                target_column=config['data']['target_column'],
                id_column=config['data'].get('id_column'),
                drop_columns=config['data'].get('drop_columns', [])
            )
            
            results['X_train'] = X_train
            results['y_train'] = y_train
            results['X_test'] = X_test
            results['test_ids'] = test_ids
            
            logger.info(f"Train shape: {X_train.shape}")
            logger.info(f"Test shape: {X_test.shape}")
            
            timings['data_loading'] = time.time() - stage_start
            logger.info(f"Data loading completed in {format_time(timings['data_loading'])}")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    # ============================================================
    # STAGE 2: PREPROCESSING
    # ============================================================
    if mode in ['full', 'train']:
        logger.info("\n" + "="*60)
        logger.info("STAGE 2: PREPROCESSING")
        logger.info("="*40)
        
        stage_start = time.time()
        
        try:
            X_train_processed, X_test_processed, preprocessor = preprocess_data(
                X_train=results['X_train'],
                X_test=results['X_test'],
                config=config,
                y_train=results['y_train']
            )
            
            results['X_train_processed'] = X_train_processed
            results['X_test_processed'] = X_test_processed
            results['preprocessor'] = preprocessor
            
            processed_train_path = config['output']['processed_train_path']
            processed_test_path = config['output']['processed_test_path']
            
            save_data(X_train_processed, processed_train_path, format='parquet')
            save_data(X_test_processed, processed_test_path, format='parquet')
            
            import joblib
            preprocessor_path = config['output']['preprocessor_path']
            ensure_dir(Path(preprocessor_path).parent)
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to: {preprocessor_path}")
            
            timings['preprocessing'] = time.time() - stage_start
            logger.info(f"Preprocessing completed in {format_time(timings['preprocessing'])}")
            
            clear_memory()
            
        except Exception as e:
            logger.error(f"Failed during preprocessing: {e}")
            raise
    
    # ============================================================
    # STAGE 3: FEATURE ENGINEERING
    # ============================================================
    if mode in ['full', 'train']:
        logger.info("\n" + "="*60)
        logger.info("STAGE 3: FEATURE ENGINEERING")
        logger.info("="*40)
        
        stage_start = time.time()
        
        try:
            X_train_fe, X_test_fe, feature_engineer = engineer_features(
                X_train=results['X_train_processed'],
                X_test=results['X_test_processed'],
                config=config
            )
            
            results['X_train_final'] = X_train_fe
            results['X_test_final'] = X_test_fe
            results['feature_engineer'] = feature_engineer
            
            logger.info(f"Final train shape: {X_train_fe.shape}")
            logger.info(f"Final test shape: {X_test_fe.shape}")
            
            if feature_engineer.created_features:
                logger.info(f"Created features: {feature_engineer.created_features}")
            
            timings['feature_engineering'] = time.time() - stage_start
            logger.info(f"Feature engineering completed in {format_time(timings['feature_engineering'])}")
            
            clear_memory()
            
        except Exception as e:
            logger.error(f"Failed during feature engineering: {e}")
            raise
    
    # ============================================================
    # STAGE 4: TRAINING
    # ============================================================
    if mode in ['full', 'train']:
        logger.info("\n" + "="*60)
        logger.info("STAGE 4: MODEL TRAINING")
        logger.info("="*40)
        
        stage_start = time.time()
        
        try:
            trainer = ModelTrainer(config=config)
            
            oof_preds, test_preds = trainer.train_cv(
                X=results['X_train_final'],
                y=results['y_train'],
                X_test=results['X_test_final']
            )
            
            results['oof_predictions'] = oof_preds
            results['test_predictions'] = test_preds
            results['trainer'] = trainer
            results['y_train_encoded'] = trainer.y_train_encoded if trainer.y_train_encoded is not None else results['y_train']
            
            trainer.save_models()
            
            timings['training'] = time.time() - stage_start
            logger.info(f"Training completed in {format_time(timings['training'])}")
            
            clear_memory()
            
        except Exception as e:
            logger.error(f"Failed during training: {e}")
            raise
    
    # ============================================================
    # STAGE 5: EVALUATION
    # ============================================================
    if mode in ['full', 'train']:
        logger.info("\n" + "="*60)
        logger.info("STAGE 5: MODEL EVALUATION")
        logger.info("="*40)
        
        stage_start = time.time()
        
        try:
            evaluator = ModelEvaluator(task_type=task_type)
            y_eval = results.get('y_train_encoded', results['y_train'])
            
            metrics = evaluator.evaluate(
                y_true=y_eval,
                y_pred=results['oof_predictions'],
                prefix='oof_'
            )
            
            results['metrics'] = metrics
            
            metrics_path = exp_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Metrics saved to: {metrics_path}")
            
            if task_type == 'classification':
                cm_path = exp_dir / "confusion_matrix.png"
                evaluator.plot_confusion_matrix(
                    y_eval,
                    results['oof_predictions'],
                    str(cm_path)
                )
                
                roc_path = exp_dir / "roc_curve.png"
                evaluator.plot_roc_curve(
                    y_eval,
                    results['oof_predictions'],
                    str(roc_path)
                )
            else:
                pred_dist_path = exp_dir / "prediction_distribution.png"
                evaluator.plot_prediction_distribution(
                    results['y_train'],
                    results['oof_predictions'],
                    str(pred_dist_path)
                )
            
            timings['evaluation'] = time.time() - stage_start
            logger.info(f"Evaluation completed in {format_time(timings['evaluation'])}")
            
        except Exception as e:
            logger.error(f"Failed during evaluation: {e}")
            raise
    
    # ============================================================
    # STAGE 6: PREDICTION & SUBMISSION
    # ============================================================
    if mode in ['full', 'predict']:
        logger.info("\n" + "="*60)
        logger.info("STAGE 6: PREDICTION & SUBMISSION")
        logger.info("="*40)
        
        stage_start = time.time()
        
        try:
            if mode == 'predict':
                from src.predictor import ModelPredictor
                model_dir = Path(config['output']['model_path']).parent
                models = ModelPredictor.load_models(str(model_dir))
                
                logger.info("Loading test data for prediction...")
                import pandas as pd
                X_test_final = pd.read_parquet(config['output']['processed_test_path'])
                test_ids = None
            else:
                models = results['trainer'].models
                X_test_final = results['X_test_final']
                test_ids = results['test_ids']
            
            submission = predict_and_submit(
                models=models,
                X_test=X_test_final,
                test_ids=test_ids,
                output_path=config['output']['submission_path'],
                model_type=model_type,
                id_column=config['data'].get('id_column', 'id'),
                target_column=config['data']['target_column']
            )
            
            results['submission'] = submission
            
            timings['prediction'] = time.time() - stage_start
            logger.info(f"Prediction completed in {format_time(timings['prediction'])}")
            
        except Exception as e:
            logger.error(f"Failed during prediction: {e}")
            raise
    
    results['timings'] = timings
    
    return results

def print_summary(results: Dict[str, Any], logger: logging.Logger, total_time: float):
    """
    Print pipeline execution summary.
    
    Args:
        results: Pipeline results dictionary
        logger: Logger instance
        total_time: Total execution time in seconds
    """
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*40)
    
    if 'timings' in results:
        logger.info("\nStage Timings:")
        for stage, duration in results['timings'].items():
            logger.info(f"  {stage:25s}: {format_time(duration)}")
    
    logger.info(f"\n{'Total Execution Time':25s}: {format_time(total_time)}")
    
    if 'metrics' in results:
        logger.info("\nModel Performance:")
        for metric_name, value in results['metrics'].items():
            logger.info(f"  {metric_name:25s}: {value:.6f}")
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completed successfully!")
    logger.info("="*60)


def main():
    """
    Main execution function.
    """
    args = parse_arguments()
    
    print("\n" + "="*60)
    print("ML PIPELINE - TABULAR DATA")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Mode: {args.mode}")
    if args.override:
        print(f"Overrides: {args.override}")
    print("="*60 + "\n")
    
    pipeline_start = time.time()
    
    try:
        # ============================================================
        # SETUP: Load Config & Initialize Logger
        # ============================================================
        
        config_path = f"config/{args.config}.yaml"
        local_config_path = "config/local.yaml"
        
        print("Loading configuration...")
        config = load_config(config_path, local_config_path)
        
        if args.override:
            print("Applying overrides...")
            config = apply_config_overrides(config, args.override)
        
        print(f"Configuration loaded: {args.config}\n")
        
        log_config = config.get('logging', {})
        if log_config.get('enabled', True):
            logger = setup_logger(
                name='main',
                log_level=log_config.get('level', 'INFO'),
                log_to_file=log_config.get('log_to_file', True),
                log_to_console=log_config.get('log_to_console', True),
                log_dir=log_config.get('log_dir', 'logs'),
                log_filename=log_config.get('log_filename', 'pipeline.log')
            )
        else:
            logger = logging.getLogger('main')
            logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            logger.addHandler(handler)
        
        logger.info("="*60)
        logger.info("ML PIPELINE EXECUTION STARTED")
        logger.info("="*60)
        logger.info(f"Experiment: {config['experiment']['name']}")
        logger.info(f"Mode: {args.mode}")
        
        seed = config['experiment']['seed']
        set_seed(seed)
        logger.info(f"Random seed set to: {seed}")
        
        log_system_info(logger)
        
        # ============================================================
        # EXECUTE PIPELINE
        # ============================================================
        
        results = run_pipeline(config, args.mode, logger)
        
        # ============================================================
        # FINALIZE: Summary & Cleanup
        # ============================================================
        
        total_time = time.time() - pipeline_start
        
        print_summary(results, logger, total_time)
        
        mlflow_config = config.get('mlflow', {})
        if mlflow_config.get('enabled', False) and MLFLOW_AVAILABLE:
            mlflow.end_run()
            logger.info("MLflow run ended")
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*60}")
        print("PIPELINE EXECUTION FAILED")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"{'='*60}\n")
        
        import traceback
        traceback.print_exc()
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.end_run()
            except:
                pass
        
        return 1


if __name__ == '__main__':
    sys.exit(main())