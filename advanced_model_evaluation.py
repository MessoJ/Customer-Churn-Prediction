import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, 
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelEvaluator:
    """
    Advanced model evaluation with comprehensive metrics and business insights.
    """
    
    def __init__(self):
        self.results = {}
        self.feature_importance = {}
        self.shap_values = {}
        
    def load_data_and_models(self):
        """Load data and trained models."""
        print("Loading data and models...")
        
        # Load data
        try:
            df = pd.read_csv('data/advanced_engineered_data.csv')
            print("Using advanced engineered data")
        except FileNotFoundError:
            df = pd.read_csv('data/engineered_data.csv')
            print("Using standard engineered data")
        
        # Load models
        models = {}
        model_files = [
            'models/overall_best_churn_model.joblib',
            'models/best_churn_model.joblib',
            'models/random_forest_model.joblib',
            'models/xgboost_model.joblib',
            'models/lightgbm_model.joblib',
            'models/voting_ensemble_model.joblib',
            'models/stacking_ensemble_model.joblib'
        ]
        
        for model_file in model_files:
            try:
                model_name = model_file.split('/')[-1].replace('_model.joblib', '')
                models[model_name] = joblib.load(model_file)
                print(f"Loaded {model_name}")
            except FileNotFoundError:
                continue
        
        # Load feature names
        try:
            feature_names = joblib.load('models/feature_names.joblib')
        except FileNotFoundError:
            feature_names = None
        
        return df, models, feature_names
    
    def prepare_data(self, df, feature_names=None):
        """Prepare data for evaluation."""
        print("Preparing data for evaluation...")
        
        # Drop customer ID if present
        if 'remainder__customerID' in df.columns:
            df = df.drop('remainder__customerID', axis=1)
        
        # Convert target to numeric
        df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
        
        # Separate features and target
        X = df.drop('remainder__Churn', axis=1)
        y = df['remainder__Churn']
        
        # Use specific features if provided
        if feature_names:
            X = X[feature_names]
        
        # Convert categorical columns to numeric
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        return X, y
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba):
        """Calculate comprehensive evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_proba),
            'average_precision': average_precision_score(y_true, y_proba)
        }
        
        # Business metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
        })
        
        return metrics
    
    def evaluate_models(self, models, X, y):
        """Evaluate all models comprehensively."""
        print("Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            print(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_comprehensive_metrics(y, y_pred, y_proba)
            results[name] = metrics
            
            # Print results
            print(f"{name.upper()} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"Average Precision: {metrics['average_precision']:.4f}")
            print(f"Specificity: {metrics['specificity']:.4f}")
            print(f"Sensitivity: {metrics['sensitivity']:.4f}")
            print("-" * 50)
        
        self.results = results
        return results
    
    def create_comprehensive_visualizations(self, models, X, y):
        """Create comprehensive visualizations for model evaluation."""
        print("Creating comprehensive visualizations...")
        
        # Create results directory
        import os
        os.makedirs('results', exist_ok=True)
        
        # 1. ROC Curves Comparison
        self.plot_roc_curves_comparison(models, X, y)
        
        # 2. Precision-Recall Curves
        self.plot_precision_recall_curves(models, X, y)
        
        # 3. Confusion Matrices
        self.plot_confusion_matrices(models, X, y)
        
        # 4. Feature Importance Analysis
        self.plot_feature_importance_analysis(models, X)
        
        # 5. SHAP Analysis
        self.create_shap_analysis(models, X, y)
        
        # 6. Business Metrics Dashboard
        self.create_business_metrics_dashboard()
        
        print("Visualizations completed and saved to results/")
    
    def plot_roc_curves_comparison(self, models, X, y):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(12, 8))
        
        for name, model in models.items():
            y_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            auc = roc_auc_score(y, y_proba)
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/roc_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self, models, X, y):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(12, 8))
        
        for name, model in models.items():
            y_proba = model.predict_proba(X)[:, 1]
            precision, recall, _ = precision_recall_curve(y, y_proba)
            avg_precision = average_precision_score(y, y_proba)
            
            plt.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self, models, X, y):
        """Plot confusion matrices for all models."""
        n_models = len(models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(models.items()):
            if idx >= len(axes):
                break
                
            y_pred = model.predict(X)
            cm = confusion_matrix(y, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name.upper()} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide empty subplots
        for idx in range(len(models), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance_analysis(self, models, X):
        """Plot feature importance analysis."""
        plt.figure(figsize=(15, 10))
        
        # Get feature importance for tree-based models
        feature_importance_data = {}
        
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importance_data[name] = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
        
        # Plot top features for each model
        n_models = len(feature_importance_data)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, importance_df) in enumerate(feature_importance_data.items()):
            if idx >= len(axes):
                break
                
            top_features = importance_df.head(10)
            sns.barplot(data=top_features, x='Importance', y='Feature', ax=axes[idx])
            axes[idx].set_title(f'{name.upper()} - Top 10 Features')
        
        # Hide empty subplots
        for idx in range(len(feature_importance_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('results/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_shap_analysis(self, models, X, y):
        """Create SHAP analysis for model interpretability."""
        print("Creating SHAP analysis...")
        
        # Use the best model for SHAP analysis
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['auc_roc'])
        best_model = models[best_model_name]
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title(f'SHAP Feature Importance - {best_model_name.upper()}')
        plt.tight_layout()
        plt.savefig('results/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Detailed SHAP plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, show=False)
        plt.title(f'SHAP Values - {best_model_name.upper()}')
        plt.tight_layout()
        plt.savefig('results/shap_detailed_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_business_metrics_dashboard(self):
        """Create a business-focused metrics dashboard."""
        print("Creating business metrics dashboard...")
        
        # Create interactive dashboard using Plotly
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Comparison', 'Business Impact Metrics',
                          'Precision vs Recall Trade-off', 'Risk Assessment'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Model performance comparison
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc_roc'] for name in model_names]
        f1_scores = [self.results[name]['f1_score'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=auc_scores, name='AUC-ROC', marker_color='blue'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=f1_scores, name='F1-Score', marker_color='red'),
            row=1, col=1
        )
        
        # Business impact metrics
        precision_scores = [self.results[name]['precision'] for name in model_names]
        recall_scores = [self.results[name]['recall'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=precision_scores, name='Precision', marker_color='green'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=recall_scores, name='Recall', marker_color='orange'),
            row=1, col=2
        )
        
        # Precision vs Recall trade-off
        fig.add_trace(
            go.Scatter(x=precision_scores, y=recall_scores, mode='markers+text',
                      text=model_names, textposition="top center",
                      marker=dict(size=10, color='purple'), name='Precision vs Recall'),
            row=2, col=1
        )
        
        # Risk assessment (False Negatives vs False Positives)
        fn_scores = [self.results[name]['false_negatives'] for name in model_names]
        fp_scores = [self.results[name]['false_positives'] for name in model_names]
        
        fig.add_trace(
            go.Bar(x=model_names, y=fn_scores, name='False Negatives', marker_color='darkred'),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Bar(x=model_names, y=fp_scores, name='False Positives', marker_color='darkblue'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Customer Churn Prediction - Business Metrics Dashboard",
            height=800,
            showlegend=True
        )
        
        fig.write_html('results/business_metrics_dashboard.html')
        print("Business metrics dashboard saved to results/business_metrics_dashboard.html")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive evaluation report."""
        print("Generating comprehensive evaluation report...")
        
        report = []
        report.append("# Customer Churn Prediction - Comprehensive Model Evaluation Report")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        
        # Find best model
        best_model = max(self.results.keys(), 
                        key=lambda x: self.results[x]['auc_roc'])
        best_metrics = self.results[best_model]
        
        report.append(f"**Best Performing Model:** {best_model.upper()}")
        report.append(f"**AUC-ROC Score:** {best_metrics['auc_roc']:.4f}")
        report.append(f"**F1-Score:** {best_metrics['f1_score']:.4f}")
        report.append(f"**Precision:** {best_metrics['precision']:.4f}")
        report.append(f"**Recall:** {best_metrics['recall']:.4f}")
        report.append("")
        
        # Business Impact Analysis
        report.append("## Business Impact Analysis")
        report.append("")
        
        total_customers = 7043  # From dataset
        churn_rate = 0.26  # From dataset
        
        for name, metrics in self.results.items():
            report.append(f"### {name.upper()}")
            report.append("")
            
            # Calculate business metrics
            predicted_churners = metrics['true_positives'] + metrics['false_positives']
            actual_churners = metrics['true_positives'] + metrics['false_negatives']
            
            report.append(f"- **Accuracy:** {metrics['accuracy']:.2%}")
            report.append(f"- **Precision:** {metrics['precision']:.2%}")
            report.append(f"- **Recall:** {metrics['recall']:.2%}")
            report.append(f"- **F1-Score:** {metrics['f1_score']:.2%}")
            report.append(f"- **AUC-ROC:** {metrics['auc_roc']:.2%}")
            report.append("")
            
            # Business interpretation
            report.append("**Business Interpretation:**")
            report.append(f"- Model correctly identifies {metrics['recall']:.1%} of actual churners")
            report.append(f"- {metrics['precision']:.1%} of predicted churners actually churn")
            report.append(f"- {metrics['false_negatives']} actual churners missed (false negatives)")
            report.append(f"- {metrics['false_positives']} loyal customers flagged as churners (false positives)")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        report.append("1. **Model Selection:** Use the model with highest AUC-ROC for production")
        report.append("2. **Threshold Tuning:** Adjust prediction threshold based on business costs")
        report.append("3. **Feature Engineering:** Focus on features with high SHAP importance")
        report.append("4. **Regular Retraining:** Retrain models monthly with new data")
        report.append("5. **A/B Testing:** Test different models in production environment")
        report.append("")
        
        # Save report
        with open('results/comprehensive_evaluation_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("Comprehensive evaluation report saved to results/comprehensive_evaluation_report.md")

def evaluate_advanced_models():
    """Main function to run advanced model evaluation."""
    # Initialize evaluator
    evaluator = AdvancedModelEvaluator()
    
    # Load data and models
    df, models, feature_names = evaluator.load_data_and_models()
    
    # Prepare data
    X, y = evaluator.prepare_data(df, feature_names)
    
    # Evaluate models
    results = evaluator.evaluate_models(models, X, y)
    
    # Create visualizations
    evaluator.create_comprehensive_visualizations(models, X, y)
    
    # Generate report
    evaluator.generate_comprehensive_report()
    
    print("Advanced model evaluation completed!")
    return results

if __name__ == "__main__":
    evaluate_advanced_models()