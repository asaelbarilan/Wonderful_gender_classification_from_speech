"""
This script explains the best-performing machine learning model using SHAP (SHapley Additive exPlanations).
SHAP provides insights into which features are most important for a model's predictions and how they contribute.
"""
import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def explain_model(model_path, data_path, output_dir):
    """
    Loads a trained model and data, computes SHAP values, and saves a summary plot.

    Args:
        model_path (str): Path to the saved model file (.joblib).
        data_path (str): Path to the test data CSV.
        output_dir (str): Directory to save the SHAP plot.
    """
    logger.info("Starting model explanation with SHAP...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the trained model
    try:
        model = joblib.load(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {model_path}. Please run the modeling pipeline first.")
        return

    # Load the test data
    try:
        test_df = pd.read_csv(data_path)
        # Separate features (X) from the target (y)
        X_test = test_df.drop(columns=['label', 'gender', 'file_path', 'speaker_id', 'split'], errors='ignore')
        logger.info(f"Successfully loaded and prepared data from {data_path}")
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Please ensure the preprocessed data exists.")
        return
    except KeyError as e:
        logger.error(f"A required column is missing from the data: {e}")
        return

    # Ensure all columns are numeric for SHAP
    X_test = X_test.select_dtypes(include='number')
    
    # Use the appropriate explainer for the model type
    # We check if the model is a tree-based model for which TreeExplainer is optimized.
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        # Use interventional feature perturbation to avoid additivity issues
        explainer = shap.TreeExplainer(model, feature_perturbation='interventional')
    elif hasattr(model, 'predict_proba'):
        # For other models like Logistic Regression or SVM, we can use KernelExplainer.
        logger.warning("Using KernelExplainer, which can be slow. TreeExplainer is preferred for tree models.")
        # We use a subset of the data for the background distribution to speed things up.
        X_test_summary = shap.kmeans(X_test, 50)
        explainer = shap.KernelExplainer(model.predict_proba, X_test_summary)
    else:
        # Fallback for models without predict_proba (should not happen with our pipeline)
        logger.error(f"Model of type {type(model).__name__} is not supported for SHAP analysis in this script.")
        return

    # Compute SHAP values
    logger.info("Calculating SHAP values... This may take a moment.")
    try:
        shap_values = explainer.shap_values(X_test)
    except Exception as e:
        logger.error(f"Error computing SHAP values: {e}")
        logger.info("Trying with check_additivity=False...")
        # Try with additivity check disabled
        shap_values = explainer.shap_values(X_test, check_additivity=False)

    # For binary classification, shap_values is a list of two arrays. We're interested in the values for the "positive" class (Female=1)
    shap_values_for_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Generate and save the SHAP summary plot
    plt.figure()
    shap.summary_plot(shap_values_for_plot, X_test, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance for Gender Classification', fontsize=16)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP summary plot saved to {plot_path}")

    # Generate and save the beeswarm plot for more detail
    plt.figure()
    shap.summary_plot(shap_values_for_plot, X_test, show=False)
    plt.title('SHAP Detailed Feature Contribution', fontsize=16)
    plt.tight_layout()
    beeswarm_plot_path = os.path.join(output_dir, "shap_beeswarm_plot.png")
    plt.savefig(beeswarm_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"SHAP beeswarm plot saved to {beeswarm_plot_path}")

if __name__ == "__main__":
    # Define paths
    MODEL_PATH = "models/best_model.joblib"
    TEST_DATA_PATH = "data/processed/final/test_data_processed.csv"
    OUTPUT_DIR = "results/machine_learning/shap_analysis"

    # Run the explanation
    explain_model(MODEL_PATH, TEST_DATA_PATH, OUTPUT_DIR) 