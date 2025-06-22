import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new features based on EDA insights.
    
    Args:
        df: DataFrame with original features
        
    Returns:
        DataFrame with additional engineered features
    """
    df_new = df.copy()
    
    # 1. MFCC Ratios (capture relative differences between coefficients)
    for i in range(1, 13):
        for j in range(i+1, 14):
            ratio = df[f'mfcc_{i}_m'] / (df[f'mfcc_{j}_m'] + 1e-8)
            # Handle infinite values
            ratio = ratio.replace([np.inf, -np.inf], np.nan)
            df_new[f'mfcc_ratio_{i}_{j}'] = ratio
    
    # 2. Energy-Pitch Interactions
    df_new['energy_pitch_interaction'] = df['log_energ'] * df['mean_pitch']
    energy_pitch_ratio = df['log_energ'] / (df['mean_pitch'] + 1e-8)
    energy_pitch_ratio = energy_pitch_ratio.replace([np.inf, -np.inf], np.nan)
    df_new['energy_pitch_ratio'] = energy_pitch_ratio
    
    # 3. Statistical Moments for Pitch
    pitch_features = ['mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch']
    df_new['pitch_range'] = df['max_pitch'] - df['min_pitch']
    pitch_variation = df['std_pitch'] / (df['mean_pitch'] + 1e-8)
    pitch_variation = pitch_variation.replace([np.inf, -np.inf], np.nan)
    df_new['pitch_variation_coef'] = pitch_variation
    
    # 4. Spectral Shape Features
    spectral_ratio = df['spectral_s'] / (df['spectral_l'] + 1e-8)
    spectral_ratio = spectral_ratio.replace([np.inf, -np.inf], np.nan)
    df_new['spectral_ratio'] = spectral_ratio
    
    spectral_energy_ratio = df['spectral_s'] / (df['log_energ'] + 1e-8)
    spectral_energy_ratio = spectral_energy_ratio.replace([np.inf, -np.inf], np.nan)
    df_new['spectral_energy_ratio'] = spectral_energy_ratio
    
    # 5. MFCC Statistics
    for i in range(1, 14):
        # Variation coefficient for each MFCC
        mfcc_var_coef = df[f'mfcc_{i}_st'] / (np.abs(df[f'mfcc_{i}_m']) + 1e-8)
        mfcc_var_coef = mfcc_var_coef.replace([np.inf, -np.inf], np.nan)
        df_new[f'mfcc_{i}_var_coef'] = mfcc_var_coef
    
    # 6. Energy-based Features
    energy_variation = df['rms_energ'] / (df['energy_en'] + 1e-8)
    energy_variation = energy_variation.replace([np.inf, -np.inf], np.nan)
    df_new['energy_variation'] = energy_variation
    
    # Normalize log energy (handle NaN in normalization)
    log_energy_mean = df['log_energ'].mean()
    log_energy_std = df['log_energ'].std()
    df_new['log_energy_normalized'] = (df['log_energ'] - log_energy_mean) / log_energy_std
    
    # Fill NaN values with median for each column (only numeric columns)
    numeric_columns = df_new.select_dtypes(include=[np.number]).columns
    df_new[numeric_columns] = df_new[numeric_columns].fillna(df_new[numeric_columns].median())
    
    return df_new

def select_uncorrelated_features(df, target_col='label', correlation_threshold=0.7, min_target_correlation=0.1):
    """
    Select features based on correlation analysis.
    
    Args:
        df: DataFrame with features
        target_col: Name of target column
        correlation_threshold: Maximum allowed correlation between features
        min_target_correlation: Minimum required correlation with target
    
    Returns:
        List of selected feature names
    """
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    
    # Get correlations with target
    target_correlations = corr_matrix[target_col].sort_values(ascending=False)
    
    # Filter features by minimum target correlation
    candidate_features = target_correlations[target_correlations > min_target_correlation].index.tolist()
    candidate_features.remove(target_col)
    
    # Remove highly correlated features
    selected_features = []
    for feature in candidate_features:
        if not selected_features:
            selected_features.append(feature)
        else:
            correlations = corr_matrix.loc[feature, selected_features]
            if correlations.max() < correlation_threshold:
                selected_features.append(feature)
    
    print(f"Selected {len(selected_features)} features:")
    for f in selected_features:
        print(f"- {f} (correlation with target: {target_correlations[f]:.3f})")
    
    return selected_features

# A custom transformer to safely apply log transform
def log_transform(x):
    # Adding a small constant to avoid log(0)
    return np.log1p(x)

def create_preprocessing_pipeline(
    log_transform_features: list,
    use_robust_scaling: bool = True,
    feature_subset: list = None
) -> Pipeline:
    """
    Creates a preprocessing pipeline with optional log transform and scaling.
    This version preserves feature names throughout the pipeline.
    """
    steps = []

    # Optional step to select a subset of features
    if feature_subset:
        steps.append(('feature_selection', FeatureSelector(feature_subset)))

    # Step to apply log transformation to specified features
    log_features_in_subset = []
    if feature_subset:
        log_features_in_subset = [f for f in log_transform_features if f in feature_subset]
    else:
        log_features_in_subset = log_transform_features
    
    if log_features_in_subset:
        steps.append(('log_transform', LogTransformer(log_features_in_subset)))

    # Step for scaling
    scaler_type = 'robust' if use_robust_scaling else 'standard'
    steps.append(('scaler', DataFrameScaler(scaler_type)))

    # Add an imputer to handle any potential NaNs
    steps.append(('imputer', DataFrameImputer(strategy='median')))

    return Pipeline(steps)

def get_feature_names(pipeline: Pipeline, input_features: list) -> list:
    """
    Returns the feature names after the pipeline has been applied.
    """
    if 'feature_selection' in pipeline.named_steps:
        return pipeline.named_steps['feature_selection'].transform(pd.DataFrame(columns=input_features)).columns.tolist()
    return input_features

# Recommended feature groups based on correlation analysis
ENERGY_FEATURES = ['rms_energ', 'energy_en', 'log_energ']
SPECTRAL_FEATURES = ['spectral_s', 'spectral_l', 'mean_spe', 'std_spe']
PITCH_FEATURES = ['mean_pitch', 'min_pitch', 'max_pitch', 'std_pitch']
MFCC_FEATURES = [f'mfcc_{i}_m' for i in range(1, 14)] + [f'mfcc_{i}_st' for i in range(1, 14)]

# Features recommended for log transform based on distribution
LOG_TRANSFORM_CANDIDATES = ENERGY_FEATURES + ['max_pitch', 'std_pitch']

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer to select features while preserving DataFrame structure."""
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        else:
            # If X is already a numpy array, assume it has the right columns
            return X

class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer to apply log transform while preserving feature names."""
    
    def __init__(self, feature_names):
        self.feature_names = feature_names
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in self.feature_names:
                if col in X_transformed.columns:
                    X_transformed[col] = np.log1p(X_transformed[col])
            return X_transformed
        else:
            # If X is already a numpy array, apply log transform
            X_transformed = X.copy()
            for i, col in enumerate(self.feature_names):
                if i < X_transformed.shape[1]:
                    X_transformed[:, i] = np.log1p(X_transformed[:, i])
            return X_transformed

class DataFrameScaler(BaseEstimator, TransformerMixin):
    """Custom scaler that preserves DataFrame structure and feature names."""
    
    def __init__(self, scaler_type='robust'):
        self.scaler_type = scaler_type
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.feature_names = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        X_transformed = self.scaler.transform(X)
        if self.feature_names is not None:
            return pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index if hasattr(X, 'index') else None)
        return X_transformed

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Custom imputer that preserves DataFrame structure and feature names."""
    
    def __init__(self, strategy='median'):
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=strategy)
        self.feature_names = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        self.imputer.fit(X)
        return self
    
    def transform(self, X):
        X_transformed = self.imputer.transform(X)
        if self.feature_names is not None:
            return pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index if hasattr(X, 'index') else None)
        return X_transformed 