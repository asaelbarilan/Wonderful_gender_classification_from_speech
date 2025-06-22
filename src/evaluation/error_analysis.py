"""
This script performs an error analysis on the model's predictions to identify patterns in misclassifications.
It compares the feature distributions of correctly classified vs. incorrectly classified samples.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_errors(model_path, data_path, output_dir):
    """
    Loads a model and test data, identifies errors, and analyzes their characteristics.

    Args:
        model_path (str): Path to the saved model file (.joblib).
        data_path (str): Path to the test data CSV.
        output_dir (str): Directory to save the error analysis plots.
    """
    logger.info("Starting error analysis...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model and data
    try:
        model = joblib.load(model_path)
        test_df = pd.read_csv(data_path)
        
        # Load selected features to ensure proper alignment
        features_file = os.path.join(os.path.dirname(data_path), 'selected_features.txt')
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                selected_features = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(selected_features)} selected features from {features_file}")
            
            # Use only the selected features for prediction
            X_test = test_df[selected_features]
        else:
            logger.warning("Selected features file not found. Using all numeric columns.")
            X_test = test_df.drop(columns=['label', 'gender', 'file_path', 'speaker_id', 'split'], errors='ignore')
            X_test = X_test.select_dtypes(include='number')
        
        y_test = test_df['label']
        
        logger.info("Successfully loaded model and data.")
        logger.info(f"X_test shape: {X_test.shape}")
        logger.info(f"Features used: {list(X_test.columns)}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}. Please run the full pipeline first.")
        return

    # Get predictions
    y_pred = model.predict(X_test)

    # Create a DataFrame with actual, predicted, and error flags
    analysis_df = X_test.copy()
    analysis_df['actual'] = y_test
    analysis_df['predicted'] = y_pred
    analysis_df['is_correct'] = (analysis_df['actual'] == analysis_df['predicted'])

    # Identify correct predictions, separated by gender
    correct_df = analysis_df[analysis_df['is_correct'] == True]
    correct_male_df = correct_df[correct_df['actual'] == 0]
    correct_female_df = correct_df[correct_df['actual'] == 1]

    # False Positive: Actual is Male (0), Predicted is Female (1)
    fp_df = analysis_df[(analysis_df['actual'] == 0) & (analysis_df['predicted'] == 1)]
    
    # False Negative: Actual is Female (1), Predicted is Male (0)
    fn_df = analysis_df[(analysis_df['actual'] == 1) & (analysis_df['predicted'] == 0)]

    if fp_df.empty and fn_df.empty:
        logger.info("Congratulations! The model made no errors on the test set.")
        return

    logger.info(f"Found {len(fp_df)} False Positives (Male predicted as Female).")
    logger.info(f"Found {len(fn_df)} False Negatives (Female predicted as Male).")

    # Analyze the distributions of features for correct vs. incorrect predictions
    # We will focus on a few key features for clarity
    key_features = [
        'mfcc_2_m', 'mfcc_13_m', 'mfcc_12_m', 'mean_pitch', 'std_pitch',
        'spectral_s', 'log_energ'
    ]
    
    # Filter out features that may not exist in the test set
    key_features = [f for f in key_features if f in analysis_df.columns]
    
    if not key_features:
        logger.error("None of the key features for error analysis were found in the data.")
        return

    num_features = len(key_features)
    num_cols = 3
    num_rows = int(np.ceil(num_features / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows * 5))
    axes = axes.flatten()

    for i, feature in enumerate(key_features):
        ax = axes[i]
        
        # Plot correct distributions for each gender
        sns.kdeplot(data=correct_male_df, x=feature, ax=ax, label='Correct Male', fill=True, color='skyblue', alpha=0.6)
        sns.kdeplot(data=correct_female_df, x=feature, ax=ax, label='Correct Female', fill=True, color='lightpink', alpha=0.6)

        # Overlay error distributions if they exist
        if not fp_df.empty:
            sns.kdeplot(data=fp_df, x=feature, ax=ax, label='False Positive (M->F)', color='darkorange', linewidth=2)
        
        if not fn_df.empty:
            sns.kdeplot(data=fn_df, x=feature, ax=ax, label='False Negative (F->M)', color='darkred', linewidth=2)
            
        ax.set_title(f'Distribution of "{feature}"', fontsize=14)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "error_analysis_detailed_distributions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Error analysis plot saved to {plot_path}")
    logger.info("Error analysis complete. Check the plot for differences in feature distributions.")

if __name__ == "__main__":
    # Define paths
    MODEL_PATH = "models/best_model.joblib"
    TEST_DATA_PATH = "data/processed/final/test_data_processed.csv"
    OUTPUT_DIR = "results/machine_learning/error_analysis"

    # Run the analysis
    analyze_errors(MODEL_PATH, TEST_DATA_PATH, OUTPUT_DIR) 