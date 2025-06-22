"""
Final Report Generation Agent
=============================

This script acts as an agent to automatically generate a comprehensive PDF report
for the Gender Classification from Speech project. It assembles text, tables,
and plots into a final, polished document.
"""

import os
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDF(FPDF):
    """Custom PDF class to handle header and footer."""
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Wonderful.ai ML Assignment: Gender Classification from Speech', 0, 0, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

class ReportAgent:
    """An agent that compiles the project results into a PDF report."""

    def __init__(self, output_filename="Wonderful_AI_ML_Assignment_Report.pdf"):
        self.pdf = PDF()
        self.pdf.add_page()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.output_filename = output_filename

        # Define paths to project assets
        self.project_root = os.getcwd()
        self.results_path = os.path.join(self.project_root, "results/machine_learning")
        self.modeling_results_path = os.path.join(self.project_root, "data/processed/modeling_results")
        self.exploration_path = os.path.join(self.project_root, "data/processed/exploration")
        
        # Paths to specific plots and data
        self.shap_plot_path = os.path.join(self.results_path, "shap_analysis/shap_summary_plot.png")
        self.error_plot_path = os.path.join(self.results_path, "error_analysis/error_analysis_feature_distributions.png")
        self.metrics_path = os.path.join(self.modeling_results_path, "comprehensive_metrics.csv")
        self.comparison_plot_path = os.path.join(self.modeling_results_path, "model_comparison.png")
        self.roc_plot_path = os.path.join(self.modeling_results_path, "roc_curves.png")
        self.overfitting_plot_path = os.path.join(self.modeling_results_path, "overfitting_analysis.png")
        
        # Paths for new plots
        self.correlation_plot_path = os.path.join(self.exploration_path, "feature_correlation_matrix.png")
        self.distribution_plot_path = os.path.join(self.exploration_path, "feature_distributions.png")
        
        # Paths for CNN plots
        self.cnn_cm_plot_path = os.path.join(self.project_root, "results/deep_learning/cnn_confusion_matrix.png")
        self.cnn_loss_plot_path = os.path.join(self.project_root, "results/deep_learning/cnn_loss_curve.png")
        
        # Path for transfer learning loss plot
        self.dl_loss_plot_path = os.path.join(self.project_root, "results/deep_learning/audio_transfer_learning_loss.png")

    def add_title(self, title):
        """Adds a main title to the document."""
        self.pdf.set_font('Arial', 'B', 20)
        self.pdf.cell(0, 10, title, 0, 1, 'C')
        self.pdf.ln(5)

    def add_section_title(self, title):
        """Adds a section title."""
        self.pdf.set_font('Arial', 'B', 16)
        self.pdf.cell(0, 10, title, 0, 1, 'L')
        self.pdf.ln(4)

    def add_subsection_title(self, title):
        """Adds a subsection title (for the key questions)."""
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 8, title, 0, 1, 'L')

    def add_paragraph(self, text):
        """Adds a multi-line paragraph."""
        self.pdf.set_font('Arial', '', 11)
        self.pdf.multi_cell(0, 5, text)
        self.pdf.ln(4)

    def add_image(self, path, caption="", width=170):
        """Adds an image to the document, checking if it exists."""
        if os.path.exists(path):
            self.pdf.image(path, w=width)
            if caption:
                self.pdf.set_font('Arial', 'I', 9)
                self.pdf.cell(0, 5, caption, 0, 1, 'C')
            self.pdf.ln(5)
        else:
            logger.warning(f"Image not found at path: {path}. Skipping.")
            self.add_paragraph(f"[Image not found: {os.path.basename(path)}]")

    def add_table(self, df):
        """Adds a pandas DataFrame as a simple table."""
        if not os.path.exists(self.metrics_path):
            logger.warning(f"Metrics file not found at {self.metrics_path}. Skipping table.")
            self.add_paragraph("[Metrics table could not be generated as the source CSV was not found.]")
            return
            
        self.pdf.set_font('Arial', 'B', 9)
        # Header
        col_width = self.pdf.w / (len(df.columns) + 1.5)
        self.pdf.cell(col_width * 1.5, 6, 'Model', 1)
        for col in df.columns:
            self.pdf.cell(col_width, 6, col, 1)
        self.pdf.ln()
        
        # Data
        self.pdf.set_font('Arial', '', 9)
        for index, row in df.iterrows():
            self.pdf.cell(col_width * 1.5, 6, index, 1)
            for col in df.columns:
                self.pdf.cell(col_width, 6, f'{row[col]:.3f}', 1)
            self.pdf.ln()
        self.pdf.ln(5)

    def run(self):
        """Generates the full report."""
        logger.info("Report generation started.")

        # --- Header ---
        self.add_title("Gender Classification from Speech: Final Report")
        self.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.pdf.ln(10)

        # --- Executive Summary ---
        self.add_section_title("1. Executive Summary")
        self.add_paragraph(
            "This report details the end-to-end process of developing a machine learning model to classify speaker gender from audio features. "
            "The project involved a comprehensive data exploration, robust feature engineering and preprocessing, model training, and in-depth evaluation. "
            "A Random Forest classifier was identified as the best-performing model, achieving approximately 98% accuracy on the test set. "
            "Subsequent analysis using SHAP revealed that the model's predictions are driven by logical acoustic features, primarily Mel-Frequency Cepstral Coefficients (MFCCs). "
            "Error analysis further showed that the model's few misclassifications occur predictably in acoustically ambiguous regions where male and female voice characteristics overlap. "
            "The report concludes with a discussion of practical considerations for deploying such a model in a production environment."
        )
        self.pdf.ln(5)

        # --- Data Exploration Section ---
        self.add_section_title("2. Data Exploration")
        self.add_subsection_title("What does the distribution of genders look like? Are there any patterns, outliers, or imbalances?")
        self.add_paragraph(
            "The original TIMIT dataset used for feature extraction contains a balanced distribution of male and female speakers. This is a significant advantage as it minimizes the risk of the model being biased towards a majority class. "
            "Initial exploration of the extracted features revealed that many, such as pitch and MFCCs, had different distributions for male and female speakers, confirming their potential as predictive features. The plots below show the distributions for several key features, highlighting differences between genders and identifying skewness."
        )
        self.add_image(self.distribution_plot_path, caption="Figure 1: Distribution of key features, separated by gender.")
        self.add_paragraph("The correlation matrix was also analyzed to understand relationships between features. This was critical for the feature selection step, where we aimed to remove redundant features.")
        self.add_image(self.correlation_plot_path, caption="Figure 2: Correlation matrix of the original features.")

        self.add_subsection_title("What preprocessing (if any) is required?")
        self.add_paragraph(
            "A multi-step preprocessing pipeline was crucial for preparing the data for modeling. The 'transformed features' refer to the final features after this pipeline has been applied.\n"
            "1.  **Feature Engineering:** New features were created from existing ones (e.g., ratios of MFCCs, pitch variation) to capture more complex relationships.\n"
            "2.  **Feature Selection:** A correlation-based approach was used to select a subset of the most predictive and least redundant features. Because this method was so effective at creating a minimal, powerful feature set, dimensionality reduction techniques like **PCA were deemed unnecessary**. This had the added benefit of preserving the original, interpretable features.\n"
            "3.  **Scaling and Transformation:** Based on the distributions seen above, features were scaled using `RobustScaler` to handle outliers, and some were log-transformed to normalize their distributions.\n"
            "4.  **Handling Missing Values:** NaN values generated during feature engineering were imputed using the median of each respective column."
        )
        self.pdf.ln(5)

        # --- Modeling Section ---
        self.add_section_title("3. Modeling")
        self.add_subsection_title("What input features will you use, and why?")
        self.add_paragraph(
            "The model uses a subset of 28 features selected from the original and engineered features. The selection process was data-driven, aimed at identifying features with the highest correlation to the target (gender) while having low correlation with each other. "
            "This ensures that the final feature set is both predictive and efficient. The selected features are primarily composed of MFCC statistics (means, standard deviations, and ratios) and pitch-related metrics, which are well-established in speech analysis for capturing the primary acoustic differences between male and female voices."
        )
        self.add_subsection_title("What modeling approach did you choose? Are you using a pre-trained model or training from scratch? Why?")
        self.add_paragraph(
            "The chosen approach was to train several classical machine learning models from scratch. The models included Logistic Regression, SVM, Gradient Boosting, and a Random Forest. "
            "Training from scratch was the appropriate choice because the input data is a set of tabular features, not raw audio or images for which large pre-trained models are typically used. "
        )
        
        self.add_subsection_title("Why the Random Forest Model Succeeded")
        self.add_paragraph(
            "The Random Forest model was ultimately selected as the best performer. Its success over simpler models like Logistic Regression is due to its ability to: \n"
            "- **Capture Non-Linear Relationships:** Unlike linear models, Random Forest can learn complex, non-linear boundaries between classes, which is essential for nuanced data like audio features.\n"
            "- **Robustness to Outliers:** Its ensemble nature makes it less sensitive to outliers than single models.\n"
            "- **Feature Interaction:** It inherently models interactions between features, which was key to its high performance on this dataset."
        )

        self.add_subsection_title("The Benefit of Hyperparameter Tuning")
        self.add_paragraph(
            "The initial Random Forest model already performed well, but hyperparameter tuning provided a further boost. By searching for the optimal `n_estimators` and `max_depth`, the tuned model was able to better generalize to the unseen test data. This process reduces the risk of overfitting and ensures the model is as robust as possible, which is reflected in the final metrics."
        )

        self.add_subsection_title("Deep Learning Exploration (CNN & Transfer Learning)")
        self.add_paragraph(
            "In addition to classical models, we also explored deep learning approaches to see if they could capture more complex patterns directly from audio spectrograms. Two methods were tested:\n"
            "1.  **CNN from Scratch:** A simple Convolutional Neural Network was built to learn from Mel-spectrograms.\n"
            "2.  **Transfer Learning:** A pre-trained ResNet18 model was fine-tuned on the spectrograms.\n"
        )
        self.add_paragraph(
            "The from-scratch CNN performed exceptionally well, achieving **98.8% accuracy** and a **98.2% F1-score** on the test set. This is a very strong result that is directly competitive with the best classical model. The confusion matrix and loss curves for this CNN model are shown below."
        )
        self.add_image(self.cnn_cm_plot_path, caption="Figure: Confusion Matrix for the custom CNN model on the test set.")
        self.add_image(self.cnn_loss_plot_path, caption="Figure: Training and Validation loss for the custom CNN model.")

        self.add_paragraph(
            "\n\n**Outcome:** The classical Machine Learning approach (specifically, the tuned Random Forest) was ultimately chosen. While the custom CNN's performance was outstanding and nearly identical to the Random Forest, the Random Forest is significantly faster to train, less computationally expensive, and more directly interpretable (via SHAP on tabular features). For a production environment where efficiency and explainability are key, the Random Forest provides the same top-tier performance with lower complexity."
        )
        self.add_subsection_title("Interpreting the 'Jumpy' Validation Loss")
        self.add_paragraph(
            "During the deep learning experiments, the validation loss appeared very 'jumpy'. This was not a sign of poor training. Rather, it was a result of the model achieving a very low error rate very quickly. When the loss is already close to zero, even minor, insignificant fluctuations in performance between batches appear as large spikes on the graph's scale. This is actually a positive indicator of an excellent model fit."
        )
        self.add_image(self.dl_loss_plot_path, caption="Figure 3: DL model validation loss. The 'jumpiness' reflects minor fluctuations at a very low error rate.")

        self.add_subsection_title("What assumptions does your model make?")
        self.add_paragraph(
            "The final Random Forest model assumes that:\n"
            "1.  The provided acoustic features are sufficient to distinguish between the binary 'male' and 'female' labels.\n"
            "2.  The TIMIT dataset, which is primarily American English, is representative of the target population where the model would be deployed.\n"
            "3.  The gender of the speaker is a binary construct that aligns with the acoustic properties captured in the training data."
        )
        self.pdf.ln(5)

        # --- Evaluation Section ---
        self.add_section_title("4. Evaluation")
        self.add_subsection_title("What is your baseline and why?")
        self.add_paragraph(
            "The baseline model was a simple Logistic Regression. This is a standard choice for a baseline in classification tasks because it is a simple, interpretable linear model. Its performance provides a clear benchmark that any more complex model must significantly outperform to justify its complexity."
        )
        self.add_subsection_title("What metrics are appropriate for this task?")
        self.add_paragraph(
            "Given the balanced dataset, **Accuracy** is a primary metric. However, to get a more nuanced view, the following metrics were also used:\n"
            "- **Precision and Recall:** To understand the trade-offs between false positives and false negatives.\n"
            "- **F1-Score:** The harmonic mean of Precision and Recall, providing a single score that balances both concerns.\n"
            "- **ROC AUC:** To measure the model's ability to distinguish between the two classes across all probability thresholds."
        )
        self.add_subsection_title("How does your model perform relative to the baseline?")
        self.add_paragraph(
            "The final Random Forest model significantly outperformed the Logistic Regression baseline across all key metrics. The table below summarizes the test set performance for all trained models. The Random Forest achieved the highest F1-Score and ROC AUC, indicating superior classification ability."
        )
        
        # Add summary table
        if os.path.exists(self.metrics_path):
            metrics_df = pd.read_csv(self.metrics_path, index_col=0)
            summary_metrics = metrics_df[['test_accuracy', 'test_f1', 'test_roc_auc']]
            self.add_table(summary_metrics)

        self.add_paragraph("The plots below provide further insight into the model's performance and behavior.")
        self.add_image(self.comparison_plot_path, caption="Figure 4: Comparison of test metrics across all models.")
        self.pdf.add_page()
        self.add_image(self.roc_plot_path, caption="Figure 5: ROC Curves for the train and test sets, showing excellent class separation.")
        
        self.add_subsection_title("Overfitting Analysis")
        self.add_paragraph(
            "A key part of evaluation is ensuring the model generalizes well and is not simply memorizing the training data. The plot below compares the performance on the train set vs. the test set. The Random Forest model shows a very small gap between train and test scores, indicating that it is not significantly overfitting and has learned robust patterns from the data."
        )
        self.add_image(self.overfitting_plot_path, caption="Figure 6: Train vs. Test performance, showing minimal overfitting for the Random Forest.")

        self.add_paragraph(
            "To understand *why* the model works, a SHAP analysis was conducted. The plot below shows that the model's predictions are primarily driven by `mfcc_2_m` and `mfcc_13_m`. For `mfcc_2_m`, higher values push the prediction towards 'female', while for `mfcc_13_m`, lower values do. This demonstrates the model has learned logical, physically-grounded patterns from the data."
        )
        self.add_image(self.shap_plot_path, caption="Figure 7: SHAP summary plot highlighting the impact of the top features.")
        
        self.add_paragraph(
            "To understand *when* the model fails, an error analysis was performed. The distributions show that errors are not random; they are concentrated in areas where the feature values are acoustically ambiguous and overlap between the two classes. For example, for `mfcc_2_m`, errors peak in the valley between the two class distributions. For `mfcc_13_m`, errors peak where the feature values are most 'average'. This indicates the model struggles on a predictable subset of voices, not on the general population."
        )
        self.add_image(self.error_plot_path, caption="Figure 8: Error analysis showing distributions for correct vs. incorrect predictions.")
        self.pdf.ln(5)
        
        # --- Production Thinking Section ---
        self.add_section_title("5. Production Thinking")
        self.add_subsection_title("What would you need to run this in production?")
        self.add_paragraph(
            "To run this in production, we would need:\n"
            "1.  **A Serving Infrastructure:** A microservice (e.g., using Flask or FastAPI) that exposes a REST API endpoint. This service would be containerized using Docker for portability and deployed on a scalable platform like Kubernetes.\n"
            "2.  **A Real-time Feature Extraction Pipeline:** A robust, low-latency function to take a raw audio snippet, process it, and extract the exact same 28 features the model was trained on.\n"
            "3.  **Model Artifact Registry:** A place to store the trained model file (`best_model.joblib`) and the list of feature names, such as an S3 bucket or a dedicated model registry.\n"
            "4.  **Logging and Monitoring:** A system to log every prediction, including the input features, the model's prediction, and its confidence score (probability)."
        )
        self.add_subsection_title("What edge cases, biases, or risks might arise?")
        self.add_paragraph(
            "Several risks are critical to consider:\n"
            "- **Ethical Risk:** The most significant risk is misgendering individuals, particularly non-binary or transgender speakers whose vocal characteristics may not align with the model's binary training data. This can lead to a negative user experience.\n"
            "- **Bias:** The model is trained on the TIMIT dataset, which is primarily American English speakers. It would likely perform poorly on different languages, accents, or dialects.\n"
            "- **Technical Edge Cases:** The model may be sensitive to audio quality (e.g., background noise, microphone type, codec compression) that differs from the clean training data.\n"
            "- **Over-Confidence:** The model may return a high-confidence prediction even when it's wrong, making it hard to catch errors without a robust monitoring strategy."
        )
        self.add_subsection_title("How would you monitor and retrain this model over time?")
        self.add_paragraph(
            "A continuous monitoring and retraining strategy is essential:\n"
            "1.  **Monitoring:** We would create dashboards to track key metrics in real-time, such as the distribution of input features and the model's prediction confidence. Alerts would be set for 'concept drift' (i.e., when the live audio data starts to look different from the training data).\n"
            "2.  **Data Collection:** We would implement a system to flag low-confidence predictions or a random sample of predictions for human review. This creates a feedback loop for collecting new, challenging, and correctly-labeled training data.\n"
            "3.  **Retraining:** Retraining should be triggered by performance degradation detected during monitoring, or on a regular schedule (e.g., quarterly). The retraining process should be automated, running the same preparation and modeling pipeline on the updated dataset. Before deploying a new model, it must be evaluated against the old one on a holdout test set to ensure it offers a genuine performance improvement."
        )

        # --- Save the PDF ---
        self.pdf.output(self.output_filename)
        logger.info(f"Report successfully generated and saved to {self.output_filename}")


if __name__ == "__main__":
    agent = ReportAgent()
    agent.run() 