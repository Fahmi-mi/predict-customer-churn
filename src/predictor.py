import pandas as pd
import numpy as np
from typing import List, Any, Optional
import logging
from pathlib import Path
import joblib


logger = logging.getLogger(__name__)


class ModelPredictor:
    """
    Universal model predictor for making predictions and generating submissions.
    """
    
    def __init__(self, models: List[Any], model_type: str = 'lightgbm'):
        """
        Initialize predictor.
        
        Args:
            models: List of trained models (from CV folds)
            model_type: Type of models ('lightgbm', 'catboost', 'xgboost')
        """
        self.models = models
        self.model_type = model_type
        
        logger.info(f"ModelPredictor initialized with {len(models)} models")
    
    def predict(self, X: pd.DataFrame, average: bool = True) -> np.ndarray:
        """
        Make predictions using ensemble of models.
        
        Args:
            X: Features for prediction
            average: Whether to average predictions from all models
        
        Returns:
            Predictions array
        """
        logger.info(f"Making predictions on {len(X)} samples")
        
        if average:
            predictions = np.zeros(len(X))
            
            for i, model in enumerate(self.models, 1):
                preds = self._predict_single_model(model, X)
                predictions += preds
                logger.debug(f"  Model {i}/{len(self.models)} predicted")
            
            predictions /= len(self.models)
            logger.info("Averaged predictions from all models")
        else:
            predictions = self._predict_single_model(self.models[0], X)
            logger.info("Used single model for prediction")
        
        return predictions
    
    def _predict_single_model(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with single model.
        
        Args:
            model: Trained model
            X: Features
        
        Returns:
            Predictions array
        """
        if self.model_type == 'lightgbm':
            import lightgbm as lgb
            preds = model.predict(X, num_iteration=model.best_iteration)
        
        elif self.model_type == 'catboost':
            preds = model.predict(X, prediction_type='Probability')
            if len(preds.shape) > 1:
                preds = preds[:, 1]
        
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            dtest = xgb.DMatrix(X)
            preds = model.predict(dtest, iteration_range=(0, model.best_iteration))
        
        else:
            preds = model.predict(X)
        
        return preds
    
    def create_submission(
        self,
        predictions: np.ndarray,
        ids: Optional[pd.Series],
        output_path: str,
        id_column: str = 'id',
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """
        Create submission file.
        
        Args:
            predictions: Prediction array
            ids: ID column (optional)
            output_path: Output file path
            id_column: Name of ID column
            target_column: Name of target column
        
        Returns:
            Submission DataFrame
        """
        logger.info("Creating submission file...")
        
        if ids is not None:
            submission = pd.DataFrame({
                id_column: ids,
                target_column: predictions
            })
        else:
            submission = pd.DataFrame({
                id_column: range(len(predictions)),
                target_column: predictions
            })
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_file, index=False)
        
        logger.info(f"Submission file saved to: {output_path}")
        logger.info(f"Submission shape: {submission.shape}")
        logger.info(f"Preview:\n{submission.head()}")
        
        return submission
    
    @staticmethod
    def load_models(model_dir: str) -> List[Any]:
        """
        Load models from directory.
        
        Args:
            model_dir: Directory containing model files
        
        Returns:
            List of loaded models
        """
        model_path = Path(model_dir)
        model_files = sorted(model_path.glob('model_fold_*.pkl'))
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        models = []
        for model_file in model_files:
            model = joblib.load(model_file)
            models.append(model)
            logger.info(f"Loaded model from: {model_file}")
        
        logger.info(f"Loaded {len(models)} models")
        return models


def predict_and_submit(
    models: List[Any],
    X_test: pd.DataFrame,
    test_ids: Optional[pd.Series],
    output_path: str,
    model_type: str = 'lightgbm',
    id_column: str = 'id',
    target_column: str = 'target'
) -> pd.DataFrame:
    """
    Convenience function to predict and create submission.
    
    Args:
        models: List of trained models
        X_test: Test features
        test_ids: Test IDs
        output_path: Submission file path
        model_type: Model type
        id_column: ID column name
        target_column: Target column name
    
    Returns:
        Submission DataFrame
    """
    predictor = ModelPredictor(models, model_type)
    predictions = predictor.predict(X_test, average=True)
    submission = predictor.create_submission(
        predictions, test_ids, output_path, id_column, target_column
    )
    
    return submission