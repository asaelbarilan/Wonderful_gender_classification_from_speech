"""
Comprehensive modeling pipeline for gender classification from speech features.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
import joblib
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
# import xgboost as xgb
# from xgboost import XGBClassifier

from src.utils.config import FIGURE_SIZE, DPI
from src.utils.helpers import setup_plotting

logger = logging.getLogger(__name__)

class GenderClassificationPipeline:
    """
    Comprehensive pipeline for gender classification modeling.
    """
    
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, 
                 X_test: pd.DataFrame, y_test: pd.Series):
        """
        Initialize the modeling pipeline.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
        """
        # Check for and handle any remaining NaN values
        if X_train.isnull().any().any():
            logger.warning("Found NaN values in training data. Filling with median...")
            X_train = X_train.fillna(X_train.median())
        
        if X_test.isnull().any().any():
            logger.warning("Found NaN values in test data. Filling with median...")
            X_test = X_test.fillna(X_test.median())
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = {}
        self.results = {}
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup plotting configuration."""
        setup_plotting()
        
    def train_baseline_models(self) -> Dict[str, Any]:
        """
        Train baseline models for comparison.
        
        Returns:
            Dictionary with model results
        """
        logger.info("Training baseline models...")
        
        baseline_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42, probability=True),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in baseline_models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions on both train and test sets
            y_train_pred = model.predict(self.X_train)
            y_train_pred_proba = model.predict_proba(self.X_train)[:, 1]
            
            y_test_pred = model.predict(self.X_test)
            y_test_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics for both sets
            train_metrics = self._calculate_metrics(self.y_train, y_train_pred, y_train_pred_proba)
            test_metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_pred_proba)
            
            # Store results
            self.models[name] = model
            results[name] = {
                'model': model,
                'train_predictions': y_train_pred,
                'train_probabilities': y_train_pred_proba,
                'train_metrics': train_metrics,
                'test_predictions': y_test_pred,
                'test_probabilities': y_test_pred_proba,
                'test_metrics': test_metrics
            }
            
            logger.info(f"{name}:")
            logger.info(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
            logger.info(f"  Test  - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Per-class metrics
        metrics['precision_male'] = precision_score(y_true, y_pred, pos_label=0)
        metrics['recall_male'] = recall_score(y_true, y_pred, pos_label=0)
        metrics['f1_male'] = f1_score(y_true, y_pred, pos_label=0)
        
        metrics['precision_female'] = precision_score(y_true, y_pred, pos_label=1)
        metrics['recall_female'] = recall_score(y_true, y_pred, pos_label=1)
        metrics['f1_female'] = f1_score(y_true, y_pred, pos_label=1)
        
        # Additional metrics
        metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0)  # True Negative Rate
        metrics['sensitivity'] = recall_score(y_true, y_pred, pos_label=1)  # True Positive Rate
        
        # Calculate confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Derived metrics
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def perform_cross_validation(self, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation for all models.
        
        Args:
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with CV results
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Cross-validating {name}...")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=cv, scoring='f1', n_jobs=-1
            )
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            logger.info(f"{name} - CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, model_name: str = 'Random Forest') -> Any:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of the model to tune
            
        Returns:
            Best model after tuning
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        if model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestClassifier(random_state=42)
            
        elif model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingClassifier(random_state=42)
            
        elif model_name == 'SVM':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
            model = SVC(random_state=42, probability=True)
            
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(self.X_train, self.y_train)
        
        # Update the model
        self.models[f"{model_name} (Tuned)"] = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def plot_model_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Plot comparison of model performances (test set).
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            model_names = list(self.results.keys())
            # Use test_metrics for comparison
            metric_values = [self.results[name]['test_metrics'][metric] for name in model_names]
            
            bars = ax.bar(model_names, metric_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C'])
            ax.set_title(f'{metric.upper()} Comparison (Test Set)', fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        return fig
    
    def plot_confusion_matrices(self, save_path: str = None) -> plt.Figure:
        """
        Plot confusion matrices for all models (test set).
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_models = len(self.results)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, (name, result) in enumerate(self.results.items()):
            ax = axes[i]
            
            cm = confusion_matrix(self.y_test, result['test_predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name} - Test Set', fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Confusion matrices plot saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(self, save_path: str = None) -> plt.Figure:
        """
        Plot ROC curves for all models (both train and test sets).
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Test set ROC curves
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['test_probabilities'])
            auc = result['test_metrics']['roc_auc']
            
            ax1.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves - Test Set', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Train set ROC curves
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_train, result['train_probabilities'])
            auc = result['train_metrics']['roc_auc']
            
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curves - Train Set', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(self, save_path: str = None) -> plt.Figure:
        """
        Plot precision-recall curves for all models (both train and test sets).
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Test set precision-recall curves
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_test, result['test_probabilities'])
            avg_precision = result['test_metrics']['average_precision']
            
            ax1.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Add baseline (random classifier)
        baseline_precision = self.y_test.mean()
        ax1.axhline(y=baseline_precision, color='navy', linestyle='--', 
                   label=f'Random (AP = {baseline_precision:.3f})')
        
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curves - Test Set', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Train set precision-recall curves
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(self.y_train, result['train_probabilities'])
            avg_precision = result['train_metrics']['average_precision']
            
            ax2.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        # Add baseline (random classifier)
        baseline_precision = self.y_train.mean()
        ax2.axhline(y=baseline_precision, color='navy', linestyle='--', 
                   label=f'Random (AP = {baseline_precision:.3f})')
        
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curves - Train Set', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Precision-recall curves plot saved to {save_path}")
        
        return fig
    
    def get_feature_importance(self, model_name: str = 'Random Forest') -> pd.DataFrame:
        """
        Get feature importance from tree-based models.
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            DataFrame with feature importance
        """
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            logger.warning(f"{model_name} doesn't support feature importance")
            return pd.DataFrame()
    
    def generate_modeling_report(self, save_dir: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive modeling report.
        
        Args:
            save_dir: Directory to save plots and results
            
        Returns:
            Dictionary with modeling results
        """
        logger.info("Generating comprehensive modeling report...")
        
        # Train baseline models
        self.train_baseline_models()
        
        # Perform cross-validation
        cv_results = self.perform_cross_validation()
        
        # Hyperparameter tuning for best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['test_metrics']['f1'])
        self.hyperparameter_tuning(best_model_name)
        
        # Generate plots
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.plot_train_test_comparison(os.path.join(save_dir, 'train_test_comparison.png'))
            self.plot_model_comparison(os.path.join(save_dir, 'model_comparison.png'))
            self.plot_confusion_matrices(os.path.join(save_dir, 'confusion_matrices.png'))
            self.plot_roc_curves(os.path.join(save_dir, 'roc_curves.png'))
            self.plot_precision_recall_curves(os.path.join(save_dir, 'precision_recall_curves.png'))
            self.plot_overfitting_analysis(os.path.join(save_dir, 'overfitting_analysis.png'))
            self.plot_feature_importance_analysis(os.path.join(save_dir, 'feature_importance_analysis.png'))
        
        # Create detailed summary
        summary = {}
        for name, result in self.results.items():
            summary[name] = {
                'train_metrics': result['train_metrics'],
                'test_metrics': result['test_metrics'],
                'overfitting_gap': {
                    metric: result['train_metrics'][metric] - result['test_metrics'][metric]
                    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                }
            }
        
        # Print detailed results
        print("\n" + "="*80)
        print("DETAILED MODEL EVALUATION RESULTS")
        print("="*80)
        
        for name, result in summary.items():
            print(f"\n{name}:")
            print("-" * 50)
            print("Train Set:")
            for metric, value in result['train_metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            print("Test Set:")
            for metric, value in result['test_metrics'].items():
                print(f"  {metric.upper()}: {value:.4f}")
            
            print("Overfitting Gap (Train - Test):")
            for metric, gap in result['overfitting_gap'].items():
                print(f"  {metric.upper()}: {gap:+.4f}")
        
        # Print comprehensive detailed metrics
        self.print_detailed_metrics()
        
        # Create and save metrics table
        metrics_table = self.create_metrics_table()
        if save_dir:
            metrics_table.to_csv(os.path.join(save_dir, 'comprehensive_metrics.csv'))
            logger.info(f"Comprehensive metrics table saved to {save_dir}")
        
        # Print summary table
        print("\n" + "="*100)
        print("SUMMARY METRICS TABLE")
        print("="*100)
        print(metrics_table[['train_accuracy', 'test_accuracy', 'train_f1', 'test_f1', 
                            'train_roc_auc', 'test_roc_auc', 'gap_accuracy', 'gap_f1']])
        
        # Create summary report
        report = {
            'baseline_results': self.results,
            'cv_results': cv_results,
            'best_model': best_model_name,
            'summary': summary
        }
        
        logger.info("Modeling report generated successfully")
        return report
    
    def plot_train_test_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Plot comparison of train vs test performance.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            model_names = list(self.results.keys())
            train_values = [self.results[name]['train_metrics'][metric] for name in model_names]
            test_values = [self.results[name]['test_metrics'][metric] for name in model_names]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, train_values, width, label='Train', color='#2E86AB', alpha=0.8)
            bars2 = ax.bar(x + width/2, test_values, width, label='Test', color='#A23B72', alpha=0.8)
            
            ax.set_title(f'{metric.upper()} - Train vs Test', fontweight='bold')
            ax.set_ylabel(metric.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, train_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            for bar, value in zip(bars2, test_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide the last subplot
        axes[-1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Train/test comparison plot saved to {save_path}")
        
        return fig
    
    def plot_overfitting_analysis(self, save_path: str = None) -> plt.Figure:
        """
        Plot overfitting analysis showing the gap between train and test performance.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        model_names = list(self.results.keys())
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            train_values = [self.results[name]['train_metrics'][metric] for name in model_names]
            test_values = [self.results[name]['test_metrics'][metric] for name in model_names]
            
            offset = (i - 2) * width
            ax.bar(x + offset, train_values, width, label=f'{metric.upper()} (Train)', alpha=0.7)
            ax.bar(x + offset, test_values, width, label=f'{metric.upper()} (Test)', alpha=0.7)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Overfitting Analysis - Train vs Test Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Overfitting analysis plot saved to {save_path}")
        
        return fig
    
    def create_metrics_table(self) -> pd.DataFrame:
        """
        Create a comprehensive metrics comparison table.
        
        Returns:
            DataFrame with all metrics for all models
        """
        # Define the metrics to include
        key_metrics = [
            'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 
            'roc_auc', 'average_precision', 'specificity', 'sensitivity',
            'precision_male', 'recall_male', 'f1_male',
            'precision_female', 'recall_female', 'f1_female'
        ]
        
        # Create results dictionary
        results_data = {}
        
        for name, result in self.results.items():
            # Train metrics
            train_metrics = {f'train_{metric}': result['train_metrics'][metric] 
                           for metric in key_metrics if metric in result['train_metrics']}
            
            # Test metrics
            test_metrics = {f'test_{metric}': result['test_metrics'][metric] 
                          for metric in key_metrics if metric in result['test_metrics']}
            
            # Overfitting gap
            overfitting_gap = {f'gap_{metric}': 
                             result['train_metrics'].get(metric, 0) - result['test_metrics'].get(metric, 0)
                             for metric in key_metrics if metric in result['train_metrics'] and metric in result['test_metrics']}
            
            results_data[name] = {**train_metrics, **test_metrics, **overfitting_gap}
        
        # Create DataFrame
        metrics_df = pd.DataFrame(results_data).T
        
        # Round to 4 decimal places
        metrics_df = metrics_df.round(4)
        
        return metrics_df
    
    def print_detailed_metrics(self):
        """
        Print detailed metrics for all models.
        """
        print("\n" + "="*100)
        print("COMPREHENSIVE MODEL EVALUATION METRICS")
        print("="*100)
        
        for name, result in self.results.items():
            print(f"\n{name.upper()}")
            print("-" * 80)
            
            # Train metrics
            print("TRAIN SET METRICS:")
            print(f"  Accuracy:           {result['train_metrics']['accuracy']:.4f}")
            print(f"  Balanced Accuracy:  {result['train_metrics']['balanced_accuracy']:.4f}")
            print(f"  Precision:          {result['train_metrics']['precision']:.4f}")
            print(f"  Recall:             {result['train_metrics']['recall']:.4f}")
            print(f"  F1-Score:           {result['train_metrics']['f1']:.4f}")
            print(f"  ROC AUC:            {result['train_metrics']['roc_auc']:.4f}")
            print(f"  Average Precision:  {result['train_metrics']['average_precision']:.4f}")
            print(f"  Specificity:        {result['train_metrics']['specificity']:.4f}")
            print(f"  Sensitivity:        {result['train_metrics']['sensitivity']:.4f}")
            
            print(f"\n  Per-Class Metrics:")
            print(f"    Male (Class 0):")
            print(f"      Precision: {result['train_metrics']['precision_male']:.4f}")
            print(f"      Recall:    {result['train_metrics']['recall_male']:.4f}")
            print(f"      F1-Score:  {result['train_metrics']['f1_male']:.4f}")
            print(f"    Female (Class 1):")
            print(f"      Precision: {result['train_metrics']['precision_female']:.4f}")
            print(f"      Recall:    {result['train_metrics']['recall_female']:.4f}")
            print(f"      F1-Score:  {result['train_metrics']['f1_female']:.4f}")
            
            # Test metrics
            print(f"\nTEST SET METRICS:")
            print(f"  Accuracy:           {result['test_metrics']['accuracy']:.4f}")
            print(f"  Balanced Accuracy:  {result['test_metrics']['balanced_accuracy']:.4f}")
            print(f"  Precision:          {result['test_metrics']['precision']:.4f}")
            print(f"  Recall:             {result['test_metrics']['recall']:.4f}")
            print(f"  F1-Score:           {result['test_metrics']['f1']:.4f}")
            print(f"  ROC AUC:            {result['test_metrics']['roc_auc']:.4f}")
            print(f"  Average Precision:  {result['test_metrics']['average_precision']:.4f}")
            print(f"  Specificity:        {result['test_metrics']['specificity']:.4f}")
            print(f"  Sensitivity:        {result['test_metrics']['sensitivity']:.4f}")
            
            print(f"\n  Per-Class Metrics:")
            print(f"    Male (Class 0):")
            print(f"      Precision: {result['test_metrics']['precision_male']:.4f}")
            print(f"      Recall:    {result['test_metrics']['recall_male']:.4f}")
            print(f"      F1-Score:  {result['test_metrics']['f1_male']:.4f}")
            print(f"    Female (Class 1):")
            print(f"      Precision: {result['test_metrics']['precision_female']:.4f}")
            print(f"      Recall:    {result['test_metrics']['recall_female']:.4f}")
            print(f"      F1-Score:  {result['test_metrics']['f1_female']:.4f}")
            
            # Overfitting analysis
            print(f"\nOVERFITTING ANALYSIS (Train - Test):")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                gap = result['train_metrics'][metric] - result['test_metrics'][metric]
                print(f"  {metric.upper()}: {gap:+.4f}")
            
            print(f"\nCONFUSION MATRIX (Test Set):")
            tn = result['test_metrics']['true_negatives']
            fp = result['test_metrics']['false_positives']
            fn = result['test_metrics']['false_negatives']
            tp = result['test_metrics']['true_positives']
            print(f"  True Negatives:  {tn}")
            print(f"  False Positives: {fp}")
            print(f"  False Negatives: {fn}")
            print(f"  True Positives:  {tp}")
            print(f"  Total:           {tn + fp + fn + tp}")
            
            print("\n" + "="*80)
    
    def plot_feature_importance_analysis(self, save_path: str = None) -> plt.Figure:
        """
        Plot feature importance analysis for tree-based models.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get models that support feature importance
        importance_models = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_models[name] = model
        
        if not importance_models:
            logger.warning("No models with feature importance found")
            return None
        
        n_models = len(importance_models)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (name, model) in enumerate(importance_models.items()):
            if i >= 4:  # Limit to 4 plots
                break
                
            ax = axes[i]
            
            # Get feature importance
            importance = model.feature_importances_
            feature_names = self.X_train.columns
            
            # Create DataFrame and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Plot top 15 features
            top_features = importance_df.head(15)
            
            bars = ax.barh(range(len(top_features)), top_features['importance'], 
                          color='#2E86AB', alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'{name} - Top 15 Features', fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for j, (bar, importance_val) in enumerate(zip(bars, top_features['importance'])):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{importance_val:.3f}', va='center', fontsize=8)
        
        # Hide empty subplots
        for i in range(len(importance_models), 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Feature importance analysis saved to {save_path}")
        
        return fig

    def save_best_model(self, metric: str = 'f1', save_dir: str = 'models/') -> str:
        """
        Selects the best model based on a specified metric and saves it to a file.

        Args:
            metric (str): The metric to use for determining the best model (from test_metrics).
            save_dir (str): The directory to save the model file.

        Returns:
            str: The path to the saved model file.
        """
        if not self.results:
            logger.error("No models have been trained. Run train_baseline_models() first.")
            return None

        best_model_name = None
        best_score = -1

        for name, result in self.results.items():
            score = result['test_metrics'].get(metric, -1)
            if score > best_score:
                best_score = score
                best_model_name = name

        if best_model_name:
            best_model = self.models[best_model_name]
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, 'best_model.joblib')
            joblib.dump(best_model, model_path)
            logger.info(f"Best model '{best_model_name}' saved to {model_path} with a {metric} score of {best_score:.4f}")
            return model_path
        else:
            logger.error("Could not determine the best model to save.")
            return None

if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load processed data
    data_dir = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\final"
    
    # Load the consolidated processed data files
    train_df = pd.read_csv(os.path.join(data_dir, 'train_data_processed.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data_processed.csv'))

    # Load the list of selected features to ensure we use the correct columns for modeling
    features_path = os.path.join(data_dir, 'selected_features.txt')
    with open(features_path, 'r') as f:
        feature_columns = [line.strip() for line in f if line.strip()]

    # Separate features (X) from the target (y)
    X_train = train_df[feature_columns]
    y_train = train_df['label']
    
    X_test = test_df[feature_columns]
    y_test = test_df['label']
    
    # Create and run modeling pipeline
    pipeline = GenderClassificationPipeline(X_train, y_train, X_test, y_test)
    
    # Generate comprehensive report
    output_dir = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\modeling_results"
    report = pipeline.generate_modeling_report(save_dir=output_dir)
    
    # Save the best model
    pipeline.save_best_model(save_dir="models/")

    logger.info("Modeling pipeline completed successfully!")
    logger.info(f"Results saved to: {output_dir}") 