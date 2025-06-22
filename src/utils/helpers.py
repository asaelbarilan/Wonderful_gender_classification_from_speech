"""
Helper functions for the Gender Classification from Speech project.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from .config import RANDOM_SEED, PLOT_STYLE, FIGURE_SIZE, DPI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

def setup_plotting():
    """Configure matplotlib and seaborn for consistent plotting."""
    plt.style.use(PLOT_STYLE)
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = FIGURE_SIZE
    plt.rcParams['figure.dpi'] = DPI

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        logger.info(f"Loading data from {filepath}")
        data = pd.read_csv(filepath)
        logger.info(f"Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: The trained model to save
        filepath: Path where to save the model
    """
    try:
        logger.info(f"Saving model to {filepath}")
        joblib.dump(model, filepath)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    try:
        logger.info(f"Loading model from {filepath}")
        model = joblib.load(filepath)
        logger.info("Model loaded successfully")
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def create_confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, 
                                class_names: list = None, title: str = "Confusion Matrix") -> plt.Figure:
    """
    Create a confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of the classes
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    return fig

def print_classification_summary(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> None:
    """
    Print a comprehensive classification summary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    """
    print(f"\n{'='*50}")
    print(f"CLASSIFICATION SUMMARY - {model_name.upper()}")
    print(f"{'='*50}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Male', 'Female']))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True Negatives (Male correctly classified): {cm[0,0]}")
    print(f"False Positives (Male classified as Female): {cm[0,1]}")
    print(f"False Negatives (Female classified as Male): {cm[1,0]}")
    print(f"True Positives (Female correctly classified): {cm[1,1]}")

def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality checks.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_counts = {}
    for col in numeric_cols:
        infinite_counts[col] = np.isinf(df[col]).sum()
    quality_report['infinite_values'] = infinite_counts
    
    return quality_report

def print_data_quality_report(quality_report: Dict[str, Any]) -> None:
    """
    Print a formatted data quality report.
    
    Args:
        quality_report: Quality report dictionary
    """
    print(f"\n{'='*50}")
    print("DATA QUALITY REPORT")
    print(f"{'='*50}")
    
    print(f"\nDataset Shape: {quality_report['shape']}")
    print(f"Total Duplicates: {quality_report['duplicates']}")
    
    print(f"\nMissing Values:")
    for col, missing in quality_report['missing_values'].items():
        if missing > 0:
            print(f"  {col}: {missing}")
    
    print(f"\nInfinite Values:")
    for col, infinite in quality_report['infinite_values'].items():
        if infinite > 0:
            print(f"  {col}: {infinite}")
    
    print(f"\nNumeric Columns ({len(quality_report['numeric_columns'])}):")
    for col in quality_report['numeric_columns']:
        print(f"  {col}")
    
    print(f"\nCategorical Columns ({len(quality_report['categorical_columns'])}):")
    for col in quality_report['categorical_columns']:
        print(f"  {col}")

def extract_gender_from_speaker_id(speaker_id: str) -> str:
    """
    Extract gender from speaker ID (first letter represents gender).
    
    Args:
        speaker_id: Speaker ID string
        
    Returns:
        Gender ('m' or 'f')
    """
    if not speaker_id or len(speaker_id) == 0:
        raise ValueError("Speaker ID cannot be empty")
    
    gender = speaker_id[0].lower()
    if gender not in ['m', 'f']:
        raise ValueError(f"Invalid gender in speaker ID: {speaker_id}")
    
    return gender 