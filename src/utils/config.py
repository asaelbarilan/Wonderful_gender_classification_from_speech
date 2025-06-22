"""
Configuration settings for the Gender Classification from Speech project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Dataset configuration
DATASET_URL = "https://www.kaggle.com/datasets/primaryobjects/voicegender"
DATASET_NAME = "voicegender"
CSV_FILENAME = "voice.csv"

# Data processing
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature configuration
AUDIO_FEATURES = [
    'meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
    'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 'maxfun',
    'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'
]

# Target variable
TARGET_COLUMN = 'label'
GENDER_MAPPING = {'m': 0, 'f': 1}  # male: 0, female: 1

# Model parameters
BASELINE_MODEL_PARAMS = {
    'strategy': 'most_frequent'  # Simple baseline using most frequent class
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': RANDOM_SEED
}

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'random_state': RANDOM_SEED
}

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'roc_auc'
]

# Cross-validation
CV_FOLDS = 5

# Model persistence
MODEL_FILENAME = "gender_classifier_model.pkl"
BASELINE_FILENAME = "baseline_model.pkl"

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (12, 8)
DPI = 300

# Production settings
MODEL_VERSION = "1.0.0"
API_PORT = 8000
API_HOST = "0.0.0.0"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 