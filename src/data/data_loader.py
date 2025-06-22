"""
Data loading and preprocessing for the Gender Classification from Speech project.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from ..utils.config import (
    RANDOM_SEED, TEST_SIZE, VALIDATION_SIZE, AUDIO_FEATURES, 
    TARGET_COLUMN, GENDER_MAPPING, RAW_DATA_DIR, PROCESSED_DATA_DIR
)
from ..utils.helpers import load_data, check_data_quality, print_data_quality_report

logger = logging.getLogger(__name__)

class VoiceDataLoader:
    """
    Data loader for the voice gender classification dataset.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV data file
        """
        self.data_path = data_path or str(RAW_DATA_DIR / "voice.csv")
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess(self) -> pd.DataFrame:
        """
        Load and preprocess the dataset.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Loading and preprocessing data...")
        
        # Load raw data
        self.data = load_data(self.data_path)
        
        # Perform data quality checks
        quality_report = check_data_quality(self.data)
        print_data_quality_report(quality_report)
        
        # Preprocess the data
        self.data = self._preprocess_data(self.data)
        
        logger.info(f"Data preprocessing completed. Final shape: {self.data.shape}")
        return self.data
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting data preprocessing...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)
        
        # Handle outliers
        df_processed = self._handle_outliers(df_processed)
        
        # Feature engineering
        df_processed = self._engineer_features(df_processed)
        
        # Encode target variable
        df_processed = self._encode_target(df_processed)
        
        # Validate final dataset
        self._validate_processed_data(df_processed)
        
        return df_processed
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Found missing values: {missing_counts[missing_counts > 0]}")
            
            # For numeric columns, fill with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled missing values in {col} with median: {median_val}")
        else:
            logger.info("No missing values found")
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numeric features using IQR method.
        
        Args:
            df: DataFrame with potential outliers
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info("Handling outliers...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            if col == TARGET_COLUMN:
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            outlier_counts[col] = outliers
            
            if outliers > 0:
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {outliers} outliers in {col}")
        
        total_outliers = sum(outlier_counts.values())
        if total_outliers > 0:
            logger.info(f"Total outliers handled: {total_outliers}")
        else:
            logger.info("No outliers found")
            
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from existing ones.
        
        Args:
            df: DataFrame with original features
            
        Returns:
            DataFrame with additional engineered features
        """
        logger.info("Engineering additional features...")
        
        # Create ratio features
        if 'meanfreq' in df.columns and 'sd' in df.columns:
            df['freq_sd_ratio'] = df['meanfreq'] / (df['sd'] + 1e-8)
        
        if 'Q75' in df.columns and 'Q25' in df.columns:
            df['q75_q25_ratio'] = df['Q75'] / (df['Q25'] + 1e-8)
        
        # Create interaction features
        if 'meanfreq' in df.columns and 'meanfun' in df.columns:
            df['freq_fun_interaction'] = df['meanfreq'] * df['meanfun']
        
        # Create statistical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
        
        if len(numeric_cols) > 0:
            df['feature_mean'] = df[numeric_cols].mean(axis=1)
            df['feature_std'] = df[numeric_cols].std(axis=1)
        
        logger.info(f"Added {len(df.columns) - len(AUDIO_FEATURES)} engineered features")
        return df
    
    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode the target variable.
        
        Args:
            df: DataFrame with target column
            
        Returns:
            DataFrame with encoded target
        """
        logger.info("Encoding target variable...")
        
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset")
        
        # Check if target is already numeric
        if df[TARGET_COLUMN].dtype in ['int64', 'float64']:
            logger.info("Target column is already numeric")
            return df
        
        # Encode categorical target
        df[TARGET_COLUMN] = self.label_encoder.fit_transform(df[TARGET_COLUMN])
        logger.info(f"Target encoding: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
        return df
    
    def _validate_processed_data(self, df: pd.DataFrame) -> None:
        """
        Validate the processed dataset.
        
        Args:
            df: Processed DataFrame to validate
        """
        logger.info("Validating processed data...")
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            raise ValueError("Processed data still contains missing values")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).sum() > 0:
                raise ValueError(f"Column {col} contains infinite values")
        
        # Check target distribution
        target_counts = df[TARGET_COLUMN].value_counts()
        logger.info(f"Target distribution: {target_counts.to_dict()}")
        
        # Check for class imbalance
        min_class_count = target_counts.min()
        max_class_count = target_counts.max()
        imbalance_ratio = max_class_count / min_class_count
        
        if imbalance_ratio > 2:
            logger.warning(f"Class imbalance detected. Ratio: {imbalance_ratio:.2f}")
        else:
            logger.info("Classes are reasonably balanced")
        
        logger.info("Data validation completed successfully")
    
    def split_data(self, test_size: float = None, val_size: float = None) -> Tuple:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        test_size = test_size or TEST_SIZE
        val_size = val_size or VALIDATION_SIZE
        
        logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
        
        if self.data is None:
            raise ValueError("Data must be loaded before splitting")
        
        # Separate features and target
        feature_cols = [col for col in self.data.columns if col != TARGET_COLUMN]
        X = self.data[feature_cols]
        y = self.data[TARGET_COLUMN]
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_SEED, stratify=y_temp
        )
        
        # Scale features
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train),
            columns=self.X_train.columns,
            index=self.X_train.index
        )
        
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val),
            columns=self.X_val.columns,
            index=self.X_val.index
        )
        
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.X_test.columns,
            index=self.X_test.index
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train set: {self.X_train.shape}")
        logger.info(f"  Validation set: {self.X_val.shape}")
        logger.info(f"  Test set: {self.X_test.shape}")
        
        return (self.X_train, self.X_val, self.X_test, 
                self.y_train, self.y_val, self.y_test)
    
    def get_feature_names(self) -> list:
        """
        Get the list of feature names.
        
        Returns:
            List of feature names
        """
        if self.X_train is not None:
            return self.X_train.columns.tolist()
        elif self.data is not None:
            return [col for col in self.data.columns if col != TARGET_COLUMN]
        else:
            return []
    
    def save_processed_data(self, filepath: str = None) -> None:
        """
        Save the processed data to disk.
        
        Args:
            filepath: Path to save the processed data
        """
        if filepath is None:
            filepath = str(PROCESSED_DATA_DIR / "processed_voice_data.csv")
        
        if self.data is not None:
            self.data.to_csv(filepath, index=False)
            logger.info(f"Processed data saved to {filepath}")
        else:
            raise ValueError("No processed data to save") 