"""
Script to prepare the final dataset for modeling by applying feature engineering and preprocessing.
"""
import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.preprocessing import (
    engineer_features, 
    select_uncorrelated_features,
    create_preprocessing_pipeline,
    LOG_TRANSFORM_CANDIDATES
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_dataset(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    correlation_threshold: float = 0.7,
    min_target_correlation: float = 0.1
) -> None:
    """
    Prepare the final dataset by applying feature engineering and preprocessing.
    
    Args:
        input_path: Path to input CSV file
        output_dir: Directory to save processed files
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        correlation_threshold: Maximum correlation between features
        min_target_correlation: Minimum correlation with target
    """
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    
    # Step 1: Feature Engineering
    logger.info("Engineering new features...")
    df_engineered = engineer_features(df)
    
    # Log NaN check after feature engineering
    logger.info("NaN check after feature engineering:")
    nan_counts = df_engineered.isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]
    if len(nan_columns) > 0:
        logger.warning(f"Found NaN values in {len(nan_columns)} columns:")
        for col, count in nan_columns.items():
            logger.warning(f"  {col}: {count} NaN values")
    else:
        logger.info("No NaN values found after feature engineering")
    
    # Step 2: Split data before preprocessing to prevent data leakage
    logger.info("Splitting data into train/test sets...")
    X = df_engineered.drop('label', axis=1)
    y = df_engineered['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Log NaN check after train/test split
    logger.info("NaN check after train/test split:")
    train_nan_counts = X_train.isnull().sum()
    test_nan_counts = X_test.isnull().sum()
    train_nan_columns = train_nan_counts[train_nan_counts > 0]
    test_nan_columns = test_nan_counts[test_nan_counts > 0]
    
    if len(train_nan_columns) > 0:
        logger.warning(f"Training data has NaN values in {len(train_nan_columns)} columns:")
        for col, count in train_nan_columns.items():
            logger.warning(f"  {col}: {count} NaN values")
    
    if len(test_nan_columns) > 0:
        logger.warning(f"Test data has NaN values in {len(test_nan_columns)} columns:")
        for col, count in test_nan_columns.items():
            logger.warning(f"  {col}: {count} NaN values")
    
    # Step 3: Feature Selection
    logger.info("Selecting features...")
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # Remove non-numeric columns before correlation analysis
    numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
    train_data_numeric = train_data[numeric_columns]
    
    selected_features = select_uncorrelated_features(
        train_data_numeric,
        correlation_threshold=correlation_threshold,
        min_target_correlation=min_target_correlation
    )
    
    # Log NaN check for selected features
    logger.info("NaN check for selected features:")
    selected_train = X_train[selected_features]
    selected_test = X_test[selected_features]
    
    selected_train_nan = selected_train.isnull().sum()
    selected_test_nan = selected_test.isnull().sum()
    
    train_nan_selected = selected_train_nan[selected_train_nan > 0]
    test_nan_selected = selected_test_nan[selected_test_nan > 0]
    
    if len(train_nan_selected) > 0:
        logger.warning(f"Selected training features have NaN values in {len(train_nan_selected)} columns:")
        for col, count in train_nan_selected.items():
            logger.warning(f"  {col}: {count} NaN values")
    
    if len(test_nan_selected) > 0:
        logger.warning(f"Selected test features have NaN values in {len(test_nan_selected)} columns:")
        for col, count in test_nan_selected.items():
            logger.warning(f"  {col}: {count} NaN values")
    
    # Step 4: Create and fit preprocessing pipeline
    logger.info("Creating preprocessing pipeline...")
    pipeline = create_preprocessing_pipeline(
        log_transform_features=LOG_TRANSFORM_CANDIDATES,
        use_robust_scaling=True,
        feature_subset=selected_features
    )
    
    # Fit and transform training data
    logger.info("Applying preprocessing to training data...")
    # The pipeline now expects a DataFrame and outputs a NumPy array.
    X_train_processed_np = pipeline.fit_transform(X_train)
    
    # Transform test data
    logger.info("Applying preprocessing to test data...")
    X_test_processed_np = pipeline.transform(X_test)
    
    # Get feature names from the pipeline
    # The final features are the ones that went into the pipeline
    final_feature_names = selected_features

    # Convert back to DataFrames with meaningful feature names
    X_train_processed = pd.DataFrame(X_train_processed_np, columns=final_feature_names, index=X_train.index)
    X_test_processed = pd.DataFrame(X_test_processed_np, columns=final_feature_names, index=X_test.index)
    
    # Apply meaningful naming to log-transformed features
    log_transform_features_in_selected = [f for f in LOG_TRANSFORM_CANDIDATES if f in selected_features]
    
    # Create a mapping for renamed features
    feature_mapping = {}
    for feature in log_transform_features_in_selected:
        if feature in final_feature_names:
            # Create meaningful name for log-transformed features
            if feature.startswith('mfcc_'):
                # For MFCC features, add log_transform suffix
                feature_mapping[feature] = f"{feature}_log_transform"
            elif feature in ['rms_energ', 'energy_en', 'log_energ']:
                # For energy features
                feature_mapping[feature] = f"{feature}_log_transform"
            elif feature in ['max_pitch', 'std_pitch']:
                # For pitch features
                feature_mapping[feature] = f"{feature}_log_transform"
    
    # Rename features in both DataFrames
    if feature_mapping:
        X_train_processed = X_train_processed.rename(columns=feature_mapping)
        X_test_processed = X_test_processed.rename(columns=feature_mapping)
        final_feature_names = [feature_mapping.get(f, f) for f in final_feature_names]
    
    # Final NaN check to be safe
    if X_train_processed.isnull().sum().sum() > 0:
        logger.error("NaN values detected after final processing step. Check pipeline.")
        # Fallback: fill just in case
        X_train_processed = X_train_processed.fillna(X_train_processed.median())
        X_test_processed = X_test_processed.fillna(X_test_processed.median())

    # Save processed datasets
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Saving processed datasets...")
    # Combine features and target for processed files, ensuring indices are aligned
    train_processed_df = pd.concat([X_train_processed, y_train], axis=1)
    test_processed_df = pd.concat([X_test_processed, y_test], axis=1)
    
    # Add original columns for reference, ensuring indices are aligned
    train_processed_df = pd.concat([train_processed_df, X_train[['gender', 'file_path', 'speaker_id', 'split']]], axis=1)
    test_processed_df = pd.concat([test_processed_df, X_test[['gender', 'file_path', 'speaker_id', 'split']]], axis=1)
    
    train_processed_df.to_csv(os.path.join(output_dir, 'train_data_processed.csv'), index=False)
    test_processed_df.to_csv(os.path.join(output_dir, 'test_data_processed.csv'), index=False)
    
    # Save feature names for later use
    with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
        f.write('\n'.join(final_feature_names))
    
    logger.info(f"Dataset preparation completed! Files saved to {output_dir}")
    logger.info(f"Final number of features: {len(final_feature_names)}")

if __name__ == "__main__":
    # Use the exact paths provided by the user
    input_csv = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\timit_features.csv"
    output_dir = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\final"
    
    prepare_dataset(input_csv, output_dir) 