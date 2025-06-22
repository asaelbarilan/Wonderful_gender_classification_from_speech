"""
Data exploration and visualization for the Gender Classification from Speech project.
"""
import os
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import TARGET_COLUMN, AUDIO_FEATURES, FIGURE_SIZE, DPI
from src.utils.helpers import setup_plotting
from src.data.preprocessing import select_uncorrelated_features, create_preprocessing_pipeline

logger = logging.getLogger(__name__)

class VoiceDataExplorer:
    """
    Data exploration and visualization for the voice gender classification dataset.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the data explorer.
        
        Args:
            data: DataFrame to explore
        """
        self.data = data
        self.setup_plotting()
        
    def setup_plotting(self):
        """Setup plotting configuration."""
        setup_plotting()
        
    def explore_basic_info(self) -> Dict[str, Any]:
        """
        Explore basic information about the dataset.
        
        Returns:
            Dictionary with basic dataset information
        """
        logger.info("Exploring basic dataset information...")
        
        info = {
            'shape': self.data.shape,
            'columns': self.data.columns.tolist(),
            'data_types': self.data.dtypes.to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': self.data.duplicated().sum()
        }
        
        print(f"\n{'='*50}")
        print("BASIC DATASET INFORMATION")
        print(f"{'='*50}")
        print(f"Shape: {info['shape']}")
        print(f"Memory Usage: {info['memory_usage'] / 1024:.2f} KB")
        print(f"Duplicates: {info['duplicates']}")
        print(f"Columns: {len(info['columns'])}")
        
        return info
    
    def explore_target_distribution(self) -> Dict[str, Any]:
        """
        Explore the target variable distribution.
        
        Returns:
            Dictionary with target distribution information
        """
        logger.info("Exploring target variable distribution...")
        
        if TARGET_COLUMN not in self.data.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found")
        
        target_counts = self.data[TARGET_COLUMN].value_counts()
        target_percentages = self.data[TARGET_COLUMN].value_counts(normalize=True) * 100
        
        info = {
            'counts': target_counts.to_dict(),
            'percentages': target_percentages.to_dict(),
            'total_samples': len(self.data),
            'class_balance_ratio': target_counts.max() / target_counts.min()
        }
        
        print(f"\n{'='*50}")
        print("TARGET VARIABLE DISTRIBUTION")
        print(f"{'='*50}")
        print(f"Total samples: {info['total_samples']}")
        print(f"Class balance ratio: {info['class_balance_ratio']:.2f}")
        
        for class_label, count in target_counts.items():
            percentage = target_percentages[class_label]
            gender = "Male" if class_label == 0 else "Female"
            print(f"{gender}: {count} samples ({percentage:.1f}%)")
        
        return info
    
    def plot_target_distribution(self, save_path: str = None) -> plt.Figure:
        """
        Plot the target variable distribution.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Count plot
        target_counts = self.data[TARGET_COLUMN].value_counts()
        gender_labels = ['Male', 'Female']
        colors = ['#2E86AB', '#A23B72']
        
        bars = ax1.bar(gender_labels, target_counts.values, color=colors, alpha=0.7)
        ax1.set_title('Gender Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Samples')
        
        # Add value labels on bars
        for bar, count in zip(bars, target_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        percentages = self.data[TARGET_COLUMN].value_counts(normalize=True) * 100
        ax2.pie(percentages.values, labels=gender_labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Gender Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Target distribution plot saved to {save_path}")
        
        return fig
    
    def explore_feature_statistics(self) -> pd.DataFrame:
        """
        Explore statistical information about features.
        
        Returns:
            DataFrame with feature statistics
        """
        logger.info("Exploring feature statistics...")
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
        
        stats = self.data[numeric_cols].describe()
        
        # Add additional statistics
        stats.loc['skewness'] = self.data[numeric_cols].skew()
        stats.loc['kurtosis'] = self.data[numeric_cols].kurtosis()
        stats.loc['missing'] = self.data[numeric_cols].isnull().sum()
        
        print(f"\n{'='*50}")
        print("FEATURE STATISTICS")
        print(f"{'='*50}")
        print(f"Number of numeric features: {len(numeric_cols)}")
        print(f"Features with missing values: {stats.loc['missing'].sum()}")
        
        return stats
    
    def plot_feature_distributions(self, n_cols: int = 4, save_path: str = None) -> plt.Figure:
        """
        Plot distributions of all numeric features.
        
        Args:
            n_cols: Number of columns in the subplot grid
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
        
        n_features = len(numeric_cols)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, col in enumerate(numeric_cols):
            ax = axes[i]
            
            # Create histogram with KDE
            sns.histplot(data=self.data, x=col, hue=TARGET_COLUMN, 
                        bins=30, alpha=0.6, ax=ax, palette=['#2E86AB', '#A23B72'])
            ax.set_title(f'{col} Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Count')
            
            # Add legend
            if i == 0:
                ax.legend(['Male', 'Female'])
            else:
                ax.legend().remove()
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Feature distributions plot saved to {save_path}")
        
        return fig
    
    def plot_correlation_matrix(self, save_path: str = None) -> plt.Figure:
        """
        Plot correlation matrix of numeric features.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f', ax=ax)
        
        ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance_preview(self, save_path: str = None) -> plt.Figure:
        """
        Plot feature importance based on correlation with target.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
        
        # Calculate correlation with target
        correlations = []
        for col in numeric_cols:
            corr = abs(self.data[col].corr(self.data[TARGET_COLUMN]))
            correlations.append((col, corr))
        
        # Sort by correlation strength
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features, corr_values = zip(*correlations)
        colors = ['#2E86AB' if x > 0 else '#A23B72' for x in corr_values]
        
        bars = ax.barh(features, corr_values, color=colors, alpha=0.7)
        ax.set_xlabel('Absolute Correlation with Target')
        ax.set_title('Feature Importance Preview (Correlation with Target)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, corr_values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Feature importance preview saved to {save_path}")
        
        return fig
    
    def plot_boxplots_by_gender(self, features: List[str] = None, save_path: str = None) -> plt.Figure:
        """
        Plot boxplots of features grouped by gender.
        
        Args:
            features: List of features to plot (if None, use top 8 by correlation)
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if features is None:
            # Get top 8 features by correlation with target
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != TARGET_COLUMN]
            
            correlations = []
            for col in numeric_cols:
                corr = abs(self.data[col].corr(self.data[TARGET_COLUMN]))
                correlations.append((col, corr))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            features = [col for col, _ in correlations[:8]]
        
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Create boxplot
            sns.boxplot(data=self.data, x=TARGET_COLUMN, y=feature, ax=ax,
                       palette=['#2E86AB', '#A23B72'])
            ax.set_title(f'{feature} by Gender', fontsize=12, fontweight='bold')
            ax.set_xlabel('Gender (0=Male, 1=Female)')
            ax.set_ylabel(feature)
        
        # Hide empty subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
            logger.info(f"Boxplots by gender saved to {save_path}")
        
        return fig
    
    def generate_exploration_report(self, save_dir: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive exploration report with all plots and statistics.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with exploration results
        """
        logger.info("Generating comprehensive exploration report...")
        
        report = {}
        
        # Basic information
        report['basic_info'] = self.explore_basic_info()
        
        # Target distribution
        report['target_distribution'] = self.explore_target_distribution()
        
        # Feature statistics
        report['feature_statistics'] = self.explore_feature_statistics()
        
        # Generate plots
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Target distribution plot
            self.plot_target_distribution(os.path.join(save_dir, 'target_distribution.png'))
            
            # Feature distributions
            self.plot_feature_distributions(save_path=os.path.join(save_dir, 'feature_distributions.png'))
            
            # Correlation matrix
            self.plot_correlation_matrix(save_path=os.path.join(save_dir, 'correlation_matrix.png'))
            
            # Feature importance preview
            self.plot_feature_importance_preview(save_path=os.path.join(save_dir, 'feature_importance.png'))
            
            # Boxplots by gender
            self.plot_boxplots_by_gender(save_path=os.path.join(save_dir, 'boxplots_by_gender.png'))
        
        logger.info("Exploration report generated successfully")
        return report

    def analyze_class_distribution(self) -> Dict[str, Any]:
        """Detailed analysis of gender distribution"""
        dist = self.data['label'].value_counts()
        percentages = self.data['label'].value_counts(normalize=True) * 100
        
        print("\nGender Distribution:")
        print(f"Male (0): {dist[0]} samples ({percentages[0]:.1f}%)")
        print(f"Female (1): {dist[1]} samples ({percentages[1]:.1f}%)")
        print(f"Imbalance ratio: {max(percentages) / min(percentages):.2f}")
        
        return {
            'counts': dist.to_dict(),
            'percentages': percentages.to_dict(),
            'imbalance_ratio': max(percentages) / min(percentages)
        }

if __name__ == "__main__":
    import logging
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Use the exact paths provided by the user
    features_csv = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\timit_features.csv"
    output_dir = r"C:\Users\Asael\PycharmProjects\wonderful_mission\data\processed\exploration"
    
    # Load data
    logger.info(f"Loading data from {features_csv}")
    df = pd.read_csv(features_csv)
    
    # Create explorer and run analysis
    explorer = VoiceDataExplorer(df)
    
    # Generate comprehensive report
    report = explorer.generate_exploration_report(save_dir=output_dir)
    
    # Analyze class distribution
    dist_stats = explorer.analyze_class_distribution()
    
    # Select features
    selected_features = select_uncorrelated_features(
        df,
        correlation_threshold=0.7,  # Remove highly correlated features
        min_target_correlation=0.1  # Keep only predictive features
    )

    # Create preprocessing pipeline
    pipeline = create_preprocessing_pipeline(
        log_transform_features=['rms_energ', 'energy_en'],  # For skewed features
        use_robust_scaling=True,  # For outlier handling
        feature_subset=selected_features
    )
    
    logger.info("Data exploration completed successfully!")
    logger.info(f"Plots saved to: {output_dir}") 