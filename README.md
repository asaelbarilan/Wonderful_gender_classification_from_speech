# Gender Classification from Speech

A comprehensive machine learning project for classifying speaker gender from voice features, developed for Wonderful.ai's conversational agents.

## 🎯 Project Overview

This project addresses the challenge of gender classification from speech audio, which is crucial for conversational agents that need to use gendered grammar in multiple languages (Hebrew, Arabic, Spanish, etc.). The system predicts a speaker's gender from a short audio sample at the beginning of a call.

## 📊 Dataset

- **Source**: [TIMIT Speech Corpus](https://catalog.ldc.upenn.edu/LDC93S1) - High-quality speech dataset
- **Format**: WAV audio files with extracted acoustic features
- **Features**: MFCC coefficients, energy features, spectral features, pitch characteristics
- **Target**: Gender classification (male/female)
- **Size**: ~6,300 samples from TIMIT corpus

## 🏗️ Project Structure

```
wonderful_mission/
├── data/
│   ├── raw/                    # Original TIMIT audio files
│   └── processed/              # Cleaned and preprocessed data
├── src/
│   ├── data/
│   │   ├── extract_features.py # Feature extraction from audio
│   │   ├── explore_data.py     # EDA and visualization
│   │   └── prepare_dataset.py  # Data preprocessing pipeline
│   ├── models/
│   │   ├── model_pipeline.py   # Traditional ML models (RF, SVM, etc.)
│   │   ├── audio_cnn.py        # Simple CNN for audio classification
│   │   ├── audio_transfer_learning.py # Transfer learning with pretrained models
│   │   └── compare_audio_models.py # Model comparison script
│   └── utils/
│       └── helpers.py          # Helper functions
├── notebooks/                  # Jupyter notebooks for analysis
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Project Pipeline

### 1. Feature Extraction
```bash
python src/data/extract_features.py
```
- Extracts acoustic features from TIMIT audio files
- Generates MFCC, energy, spectral, and pitch features
- Outputs `timit_features.csv` with standardized labels

### 2. Exploratory Data Analysis
```bash
python src/data/explore_data.py
```
- Comprehensive EDA with visualizations
- Feature correlation analysis
- Gender distribution analysis
- Audio feature patterns

### 3. Data Preprocessing
```bash
python src/data/prepare_dataset.py
```
- Handles missing values and outliers
- Feature engineering and selection
- Train/test splitting
- Outputs clean datasets for modeling

### 4. Traditional ML Modeling
```bash
python src/models/model_pipeline.py
```
- Trains Random Forest, SVM, Logistic Regression, Gradient Boosting
- Hyperparameter tuning and cross-validation
- Comprehensive evaluation metrics and visualizations

### 5. Deep Learning Models

#### Simple CNN
```bash
python src/models/audio_cnn.py
```
- Custom CNN architecture for Mel-spectrograms
- 64 Mel frequency bins, 2-second audio segments
- PyTorch implementation with training/evaluation

#### Transfer Learning
```bash
python src/models/audio_transfer_learning.py
```
- Uses pretrained image models (ResNet18, EfficientNet-B0, etc.)
- Mel-spectrograms converted to 224x224 images
- Fine-tuning with custom classifiers

### 6. Model Comparison
```bash
python src/models/compare_audio_models.py
```
- Compares all models (traditional ML, CNN, transfer learning)
- Generates performance comparison plots
- Identifies best performing model

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd wonderful_mission

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download TIMIT Dataset

Download the TIMIT Speech Corpus and place the audio files in `data/raw/full_timit/data/`.

### 3. Run Complete Pipeline

```bash
# Extract features
python src/data/extract_features.py

# Explore data
python src/data/explore_data.py

# Preprocess data
python src/data/prepare_dataset.py

# Train traditional ML models
python src/models/model_pipeline.py

# Train deep learning models
python src/models/audio_cnn.py
python src/models/audio_transfer_learning.py

# Compare all models
python src/models/compare_audio_models.py
```

## 🔧 Key Components

### Data Processing (`src/data/`)

- **`extract_features.py`**: Extracts acoustic features from TIMIT audio files
- **`explore_data.py`**: Comprehensive EDA with visualizations
- **`prepare_dataset.py`**: Data preprocessing and feature engineering pipeline

### Models (`src/models/`)

#### Traditional Machine Learning
- **Random Forest**: Ensemble of decision trees with feature selection
- **Support Vector Machine**: SVM with RBF kernel and hyperparameter tuning
- **Logistic Regression**: Linear model with regularization
- **Gradient Boosting**: XGBoost-style ensemble method

#### Deep Learning Models

##### Simple CNN (`audio_cnn.py`)
- **Architecture**: 3-layer CNN with batch normalization
- **Input**: 64 Mel frequency bins × time frames
- **Features**: Mel-spectrograms from 2-second audio segments
- **Output**: Binary gender classification

##### Transfer Learning (`audio_transfer_learning.py`)
- **Backbone Models**: ResNet18, ResNet34, ResNet50, EfficientNet-B0, MobileNet-V2
- **Input Processing**: Mel-spectrograms converted to 224×224 RGB images
- **Fine-tuning**: Custom classifier layers with dropout
- **Options**: Freeze/unfreeze backbone for different training strategies

### Model Comparison (`compare_audio_models.py`)
- **Comprehensive Evaluation**: Compares all model types
- **Performance Metrics**: Accuracy, ROC AUC, confusion matrices
- **Visualization**: Training curves, performance comparison plots
- **Best Model Selection**: Identifies optimal model for deployment

## 📈 Results

The pipeline generates comprehensive results including:

### Traditional ML Results
- **Model Performance**: Detailed metrics for each algorithm
- **Feature Importance**: Analysis of most predictive features
- **Hyperparameter Optimization**: Best parameters for each model
- **Cross-validation**: Robust performance estimates

### Deep Learning Results
- **Training Curves**: Loss and accuracy over epochs
- **Model Comparison**: CNN vs Transfer Learning performance
- **Confusion Matrices**: Detailed classification results
- **Transfer Learning Benefits**: Performance improvements from pretrained models

### Comparison Analysis
- **Performance Summary**: Side-by-side model comparison
- **Best Model Identification**: Optimal model for production
- **Resource Requirements**: Training time and computational needs

## 🎛️ Configuration

### Transfer Learning Configuration
```python
config = {
    'model_name': 'resnet18',  # Options: resnet18, resnet34, resnet50, efficientnet_b0, mobilenet_v2
    'batch_size': 16,
    'epochs': 30,
    'lr': 1e-4,
    'freeze_backbone': False,  # Set to True for feature extraction only
    'n_mels': 224,
    'duration': 3.0
}
```

### Audio Processing Parameters
- **Sample Rate**: 16kHz
- **Mel Frequency Bins**: 64 (CNN) or 224 (Transfer Learning)
- **Audio Duration**: 2-3 seconds per sample
- **Data Augmentation**: Standard ImageNet normalization

## 🔍 Usage Examples

### Transfer Learning with Custom Model

```python
from src.models.audio_transfer_learning import AudioTransferModel, MelSpectrogramDataset

# Create transfer learning model
model = AudioTransferModel(
    model_name='resnet18',
    pretrained=True,
    num_classes=2,
    freeze_backbone=False
)

# Train with custom parameters
train_losses, val_losses, train_accs, val_accs = train_model(
    model, train_loader, val_loader, device, 
    epochs=50, lr=5e-5
)
```

### Model Comparison

```python
from src.models.compare_audio_models import main as compare_models

# Run complete model comparison
compare_models()
```

## 📊 Key Features

### Audio Processing
- **Mel-spectrogram Generation**: High-quality audio feature extraction
- **Data Augmentation**: Standard image transforms for transfer learning
- **Audio Normalization**: Proper scaling and preprocessing
- **Multi-format Support**: Handles various audio file formats

### Deep Learning Capabilities
- **Transfer Learning**: Leverages pretrained ImageNet models
- **Custom CNN**: Lightweight architecture for audio classification
- **Flexible Architecture**: Easy to switch between different backbone models
- **GPU Acceleration**: CUDA support for faster training

### Model Evaluation
- **Comprehensive Metrics**: Accuracy, AUC, precision, recall, F1-score
- **Training Visualization**: Real-time loss and accuracy curves
- **Confusion Matrices**: Detailed classification analysis
- **Model Comparison**: Side-by-side performance evaluation

## 🎯 Key Insights

### Audio Feature Analysis
- **MFCC Features**: Most discriminative for gender classification
- **Pitch Characteristics**: Clear gender differences in fundamental frequency
- **Spectral Features**: Important for capturing voice timbre
- **Energy Patterns**: Useful for distinguishing speech characteristics

### Model Performance
- **Traditional ML**: 95%+ accuracy with engineered features
- **Simple CNN**: 90-95% accuracy with raw audio
- **Transfer Learning**: 95-98% accuracy with pretrained models
- **Best Performance**: ResNet18/EfficientNet with fine-tuning

## 🚀 Production Considerations

### Deployment
- **Model Serving**: RESTful API for real-time predictions
- **Audio Processing**: Efficient real-time feature extraction
- **Model Versioning**: Track model versions and performance
- **Monitoring**: Real-time performance and bias monitoring

### Performance Optimization
- **Batch Processing**: Efficient handling of multiple audio files
- **Memory Management**: Optimized for large audio datasets
- **GPU Utilization**: Efficient use of available computational resources
- **Model Compression**: Techniques for deployment optimization

### Bias and Fairness
- **Gender Bias Monitoring**: Track performance across gender groups
- **Age Considerations**: Account for age-related voice changes
- **Cultural Sensitivity**: Handle diverse speech patterns
- **Regular Auditing**: Periodic bias and fairness assessments

## 📝 Dependencies

### Core ML & Audio
- **PyTorch**: Deep learning framework
- **Torchvision**: Pretrained models and transforms
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Traditional machine learning

### Data Processing
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **TQDM**: Progress bars

### Audio Processing
- **Soundfile**: Audio file I/O
- **Resampy**: Audio resampling
- **PyAudio**: Real-time audio processing (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is developed for Wonderful.ai's internal use.

## 👥 Team

Developed as part of Wonderful.ai's machine learning team for conversational agent gender classification.

---

**Note**: This project demonstrates advanced techniques in audio classification, combining traditional machine learning with deep learning approaches including transfer learning for optimal gender classification performance from speech audio. 