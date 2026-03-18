import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Universal model evaluator for calculating metrics and creating visualizations.
    """
    
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize evaluator.
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        logger.info(f"ModelEvaluator initialized: task={task_type}")
        
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = '',
    ) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predictions
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of metrics
        """
        if self.task_type == 'classification':
            return self._evaluate_classification(y_true, y_pred, prefix)
        else:
            return self._evaluate_regression(y_true, y_pred, prefix)
        
    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        prefix: str = '',
    ) -> Dict[str, float]:
        """
        Evaluate classification metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of metrics
        """
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        
        metrics = {
            f'{prefix}accuracy': accuracy_score(y_true, y_pred_class),
            f'{prefix}precision': precision_score(y_true, y_pred_class, zero_division=0),
            f'{prefix}recall': recall_score(y_true, y_pred_class, zero_division=0),
            f'{prefix}f1': f1_score(y_true, y_pred_class, zero_division=0),
            f'{prefix}auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        logger.info(f"\n{prefix}Classification Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.6f}")
        
        return metrics
    
    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = ''
    ) -> Dict[str, float]:
        """
        Evaluate regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            prefix: Prefix for metric names
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            f'{prefix}rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            f'{prefix}mae': mean_absolute_error(y_true, y_pred),
            f'{prefix}r2': r2_score(y_true, y_pred),
            f'{prefix}mape': mean_absolute_percentage_error(y_true, y_pred)
        }
        
        logger.info(f"\n{prefix}Regression Metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.6f}")
        
        return metrics
    
    def plot_feature_importance(
        self,
        feature_importances: pd.DataFrame,
        output_path: str,
        top_n: int = 20
    ) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with 'feature' and 'importance' columns
            output_path: Output file path
            top_n: Number of top features to plot
        """
        plt.figure(figsize=(10, 8))
        
        top_features = feature_importances.head(top_n)
        
        sns.barplot(
            data=top_features,
            y='feature',
            x='importance',
            palette='viridis'
        )
        
        plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to: {output_path}")
        
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        output_path: str
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            output_path: Output file path
        """
        y_pred_class = (y_pred_proba > 0.5).astype(int)
        
        cm = confusion_matrix(y_true, y_pred_class)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive']
        )
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to: {output_path}")
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        output_path: str
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            output_path: Output file path
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve saved to: {output_path}")
    
    def plot_prediction_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str
    ) -> None:
        """
        Plot prediction distribution (for regression).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            output_path: Output file path
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('True Values', fontsize=12)
        axes[0].set_ylabel('Predicted Values', fontsize=12)
        axes[0].set_title('True vs Predicted', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=12)
        axes[1].set_ylabel('Residuals', fontsize=12)
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Prediction distribution plot saved to: {output_path}")