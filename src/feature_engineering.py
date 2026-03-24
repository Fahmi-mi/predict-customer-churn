import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering for creating interactions, binning, and derived features.
    
    All operations are vectorized for performance.
    """
    
    def __init__(self, config:Dict[str, Any]):
        """
        Initialize feature engineer with config.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fe_config = config.get("feature_engineering", {})
        self.interactions = self.fe_config.get("interactions", [])
        self.binning = self.fe_config.get("binning", {})
        
        self.created_features: List[str] = []
        
        logger.info("FeatureEngineer initialized")
        
    def fit(self, X:pd.DataFrame, y:Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit feature engineer (currently stateless, for future extensions).
        
        Args:
            X: Training features
            y: Training target (optional)
        
        Returns:
            Self
        """
        logger.info("Fitting FeatureEngineer...")
        
        self._validate_interactions(X)
        self._validate_binning(X)
        
        logger.info("FeatureEngineer fitted")
        return self
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating new features.
        
        Args:
            X: Input features
        
        Returns:
            DataFrame with new features
        """
        if not self.fe_config.get("enabled", True):
            logger.info("Feature engineering disabled, returning original data")
            return X.copy()
        
        logger.info("Creating engineered features...")
        
        X_fe = X.copy()
        self.created_features = []
        
        if self.interactions:
            X_fe = self._create_interactions(X_fe)
            
        if self.binning:
            X_fe = self._create_bins(X_fe)
            
        logger.info(f"Created {len(self.created_features)} new features")
        logger.info(f"Final shape: {X_fe.shape}")
        
        return X_fe
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Args:
            X: Input features
            y: Target (optional)
        
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)
    
    def _validate_interactions(self, X: pd.DataFrame) -> None:
        """
        Validate interaction specifications.
        
        Args:
            X: Input DataFrame
        
        Raises:
            ValueError: If interaction spec is invalid
        """
        for interaction in self.interactions:
            parts = interaction.split('*')
            
            if len(parts) != 2:
                raise ValueError(f"Invalid interaction format: {interaction}. Use 'Feature1*Feature2'")
            
            for feature in parts:
                feature = feature.strip()
                if feature not in X.columns:
                    logger.warning(f"Feature not found for interaction: {feature}")
    
    def _validate_binning(self, X: pd.DataFrame) -> None:
        """
        Validate binning specifications.
        
        Args:
            X: Input DataFrame
        
        Raises:
            ValueError: If binning spec is invalid
        """
        for feature, bins in self.binning.items():
            if feature not in X.columns:
                logger.warning(f"Feature not found for binning: {feature}")
            
            if not isinstance(bins, list) or len(bins) < 2:
                raise ValueError(f"Invalid bins for {feature}. Must be list with at least 2 values")
    
    def _create_interactions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features (vectorized).
        
        Args:
            X: Input DataFrame
        
        Returns:
            DataFrame with interaction features
        """
        for interaction in self.interactions:
            parts = [p.strip() for p in interaction.split('*')]
            
            if len(parts) != 2:
                continue
            
            feat1, feat2 = parts
            
            if feat1 not in X.columns or feat2 not in X.columns:
                logger.warning(f"Skipping interaction {interaction}: features not found")
                continue
            
            new_feature_name = f"{feat1}_x_{feat2}"
            
            X[new_feature_name] = X[feat1] * X[feat2]
            
            self.created_features.append(new_feature_name)
            logger.debug(f"Created interaction: {new_feature_name}")
        
        return X
    
    def _create_bins(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned features (vectorized).
        
        Args:
            X: Input DataFrame
        
        Returns:
            DataFrame with binned features
        """
        for feature, bins in self.binning.items():
            if feature not in X.columns:
                logger.warning(f"Skipping binning for {feature}: feature not found")
                continue
            
            try:
                new_feature_name = f"{feature}_binned"
                # Use numeric bin indices so all downstream models receive numeric features.
                binned = pd.cut(
                    X[feature],
                    bins=bins,
                    labels=False,
                    include_lowest=True,
                    duplicates='drop'
                )

                # NaN can appear when value falls outside configured edges.
                X[new_feature_name] = binned.fillna(-1).astype(np.int16)
                
                self.created_features.append(new_feature_name)
                logger.debug(f"Created bins for {feature}: {new_feature_name}")
                
            except Exception as e:
                logger.error(f"Error creating bins for {feature}: {e}")
        
        return X
    
    def get_created_features(self) -> List[str]:
        """
        Get list of created feature names.
        
        Returns:
            List of feature names
        """
        return self.created_features


def engineer_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    config: Dict[str, Any]
) -> tuple:
    """
    Convenience function to engineer features for train and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        config: Configuration dictionary
    
    Returns:
        Tuple of (X_train_engineered, X_test_engineered, feature_engineer)
    """
    engineer = FeatureEngineer(config)
    
    X_train_engineered = engineer.fit_transform(X_train)
    X_test_engineered = engineer.transform(X_test)
    
    return X_train_engineered, X_test_engineered, engineer