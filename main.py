"""
Main execution script for the Gender Classification from Speech project.
"""
import logging
import sys
import os
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import VoiceDataLoader
from src.data.data_exploration import VoiceDataExplorer
from src.models.baseline import BaselineModel, RuleBasedBaseline, BaselineEnsemble
from src.models.ml_model import GenderClassifier, ModelEnsemble, hyperparameter_tuning
from src.evaluation.metrics import ModelEvaluator
from src.utils.config import RESULTS_DIR, MODELS_DIR
from src.utils.helpers import save_model, setup_plotting
from src.data.extract_timit_features import process_timit_directory

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gender_classification.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_timit_feature_extraction(timit_root, output_csv):
    logger.info("="*60)
    logger.info("STEP 0: TIMIT FEATURE EXTRACTION")
    logger.info("="*60)
    logger.info(f"Extracting features from TIMIT dataset at {timit_root}...")
    process_timit_directory(timit_root, output_csv)
    logger.info(f"Feature extraction complete. Features saved to {output_csv}")

def main():
    """
    Main execution function for the gender classification pipeline.
    """
    logger.info("Starting Gender Classification from Speech Pipeline")
    
    # Optionally run feature extraction first
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract_features', action='store_true', help='Run TIMIT feature extraction step')
    parser.add_argument('--timit_root', type=str, default='data/raw/full_timit/data', help='Path to TIMIT data/TRAIN and data/TEST directory')
    parser.add_argument('--output_csv', type=str, default='data/processed/timit_features.csv', help='Path to output CSV file')
    args, unknown = parser.parse_known_args()

    if args.extract_features:
        run_timit_feature_extraction(args.timit_root, args.output_csv)
        # Optionally exit after extraction
        return

    try:
        # Step 1: Data Loading and Preprocessing
        logger.info("="*60)
        logger.info("STEP 1: DATA LOADING AND PREPROCESSING")
        logger.info("="*60)
        
        data_loader = VoiceDataLoader()
        data = data_loader.load_and_preprocess()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data()
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Validation: {X_val.shape}")
        logger.info(f"  Test: {X_test.shape}")
        
        # Step 2: Data Exploration
        logger.info("="*60)
        logger.info("STEP 2: DATA EXPLORATION")
        logger.info("="*60)
        
        explorer = VoiceDataExplorer(data)
        exploration_report = explorer.generate_exploration_report(
            save_dir=str(RESULTS_DIR / "exploration")
        )
        
        # Step 3: Baseline Models
        logger.info("="*60)
        logger.info("STEP 3: BASELINE MODELS")
        logger.info("="*60)
        
        # Create baseline models
        baseline_models = {
            'most_frequent': BaselineModel('most_frequent'),
            'stratified': BaselineModel('stratified'),
            'rule_based': RuleBasedBaseline(),
            'ensemble': BaselineEnsemble()
        }
        
        baseline_results = {}
        
        for name, model in baseline_models.items():
            logger.info(f"Training and evaluating {name} baseline...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            if name == 'ensemble':
                results = model.evaluate(X_val, y_val)
                baseline_results[name] = results
            else:
                results = model.evaluate(X_val, y_val)
                baseline_results[name] = results
            
            # Save model
            save_model(model, str(MODELS_DIR / f"baseline_{name}.pkl"))
        
        # Step 4: Machine Learning Models
        logger.info("="*60)
        logger.info("STEP 4: MACHINE LEARNING MODELS")
        logger.info("="*60)
        
        # Create ML models
        ml_models = {
            'random_forest': GenderClassifier('random_forest'),
            'svm': GenderClassifier('svm'),
            'logistic': GenderClassifier('logistic')
        }
        
        ml_results = {}
        
        for name, model in ml_models.items():
            logger.info(f"Training and evaluating {name} model...")
            
            # Train model with feature selection
            model.fit(X_train, y_train, feature_selection=True, n_features=15)
            
            # Evaluate on validation set
            results = model.evaluate(X_val, y_val)
            ml_results[name] = results
            
            # Save model
            save_model(model, str(MODELS_DIR / f"ml_{name}.pkl"))
        
        # Step 5: Model Ensemble
        logger.info("="*60)
        logger.info("STEP 5: MODEL ENSEMBLE")
        logger.info("="*60)
        
        # Create ensemble of best performing models
        ensemble_models = [
            GenderClassifier('random_forest'),
            GenderClassifier('svm'),
            GenderClassifier('logistic')
        ]
        
        ensemble = ModelEnsemble(ensemble_models)
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_results = ensemble.evaluate(X_val, y_val)
        ml_results['ensemble'] = ensemble_results
        
        # Save ensemble
        save_model(ensemble, str(MODELS_DIR / "ensemble.pkl"))
        
        # Step 6: Comprehensive Evaluation
        logger.info("="*60)
        logger.info("STEP 6: COMPREHENSIVE EVALUATION")
        logger.info("="*60)
        
        evaluator = ModelEvaluator()
        
        # Evaluate all models on test set
        all_results = {}
        
        # Test baseline models
        for name, model in baseline_models.items():
            logger.info(f"Evaluating {name} baseline on test set...")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'get_feature_importance'):
                try:
                    feature_importance = model.get_feature_importance()
                except:
                    pass
            
            report = evaluator.generate_evaluation_report(
                y_test.values, y_pred, y_proba, 
                f"Baseline_{name.replace('_', ' ').title()}",
                feature_importance,
                str(RESULTS_DIR / "evaluation" / "baselines")
            )
            all_results[f"baseline_{name}"] = report
        
        # Test ML models
        for name, model in ml_models.items():
            logger.info(f"Evaluating {name} model on test set...")
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            feature_importance = model.get_feature_importance()
            
            report = evaluator.generate_evaluation_report(
                y_test.values, y_pred, y_proba,
                f"ML_{name.replace('_', ' ').title()}",
                feature_importance,
                str(RESULTS_DIR / "evaluation" / "ml_models")
            )
            all_results[f"ml_{name}"] = report
        
        # Test ensemble
        logger.info("Evaluating ensemble on test set...")
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        
        report = evaluator.generate_evaluation_report(
            y_test.values, y_pred, y_proba,
            "Model_Ensemble",
            None,  # Ensemble doesn't have single feature importance
            str(RESULTS_DIR / "evaluation" / "ensemble")
        )
        all_results["ensemble"] = report
        
        # Step 7: Model Comparison
        logger.info("="*60)
        logger.info("STEP 7: MODEL COMPARISON")
        logger.info("="*60)
        
        # Compare all models
        comparison_results = evaluator.compare_models(
            all_results,
            str(RESULTS_DIR / "comparison")
        )
        
        # Step 8: Final Report
        logger.info("="*60)
        logger.info("STEP 8: GENERATING FINAL REPORT")
        logger.info("="*60)
        
        # Find best model
        best_model_name = None
        best_accuracy = 0
        
        for model_name, results in all_results.items():
            accuracy = results['metrics']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        logger.info(f"Best performing model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        # Generate summary report
        generate_summary_report(all_results, comparison_results, best_model_name)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Results saved to: {RESULTS_DIR}")
        logger.info(f"Models saved to: {MODELS_DIR}")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

def generate_summary_report(all_results, comparison_results, best_model_name):
    """
    Generate a summary report of all results.
    
    Args:
        all_results: Dictionary with all model results
        comparison_results: Model comparison results
        best_model_name: Name of the best performing model
    """
    report_path = RESULTS_DIR / "summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GENDER CLASSIFICATION FROM SPEECH - SUMMARY REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Best Accuracy: {all_results[best_model_name]['metrics']['accuracy']:.4f}\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 40 + "\n")
        
        # Sort models by accuracy
        sorted_models = sorted(
            all_results.items(), 
            key=lambda x: x[1]['metrics']['accuracy'], 
            reverse=True
        )
        
        for model_name, results in sorted_models:
            metrics = results['metrics']
            f.write(f"{model_name}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1_score']:.4f}\n")
            if metrics.get('roc_auc'):
                f.write(f"  ROC AUC:   {metrics['roc_auc']:.4f}\n")
            f.write("\n")
        
        f.write("BEST MODELS BY METRIC\n")
        f.write("-" * 40 + "\n")
        
        best_models = comparison_results['best_models']
        for metric, (model_name, score) in best_models.items():
            f.write(f"{metric.title()}: {model_name} ({score:.4f})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    logger.info(f"Summary report saved to: {report_path}")

if __name__ == "__main__":
    main() 