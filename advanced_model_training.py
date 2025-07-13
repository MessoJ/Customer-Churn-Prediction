import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    RandomizedSearchCV, GridSearchCV
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    """
    Advanced model training with state-of-the-art techniques for customer churn prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.feature_names = None
        self.scaler = RobustScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_and_prepare_data(self):
        """Load and prepare data for advanced modeling."""
        print("Loading and preparing data...")
        
        # Try to load advanced engineered data first
        try:
            df = pd.read_csv('data/advanced_engineered_data.csv')
            print("Using advanced engineered data")
        except FileNotFoundError:
            df = pd.read_csv('data/engineered_data.csv')
            print("Using standard engineered data")
        
        # Drop customer ID if present
        if 'remainder__customerID' in df.columns:
            df = df.drop('remainder__customerID', axis=1)
        
        # Convert target to numeric
        df['remainder__Churn'] = df['remainder__Churn'].map({'Yes': 1, 'No': 0})
        
        # Separate features and target
        X = df.drop('remainder__Churn', axis=1)
        y = df['remainder__Churn']
        
        # Convert categorical columns to numeric
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]
        
        # Handle missing values
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def create_ensemble_models(self):
        """Create a diverse set of models for ensemble learning."""
        print("Creating ensemble models...")
        
        # Base models
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                verbose=-1
            ),
            'catboost': cb.CatBoostClassifier(
                iterations=200, learning_rate=0.1, depth=6,
                random_state=42, verbose=False
            ),
            'logistic_regression': LogisticRegression(
                C=1.0, random_state=42, max_iter=1000
            ),
            'svm': SVC(probability=True, random_state=42)
        }
        
        return self.models
    
    def handle_class_imbalance(self, X, y):
        """Handle class imbalance using advanced techniques."""
        print("Handling class imbalance...")
        
        # Check class distribution
        print(f"Original class distribution: {np.bincount(y)}")
        
        # Create balanced datasets using different techniques
        balanced_datasets = {}
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        balanced_datasets['smote'] = (X_smote, y_smote)
        
        # ADASYN
        adasyn = ADASYN(random_state=42)
        X_adasyn, y_adasyn = adasyn.fit_resample(X, y)
        balanced_datasets['adasyn'] = (X_adasyn, y_adasyn)
        
        # SMOTEENN
        smoteenn = SMOTEENN(random_state=42)
        X_smoteenn, y_smoteenn = smoteenn.fit_resample(X, y)
        balanced_datasets['smoteenn'] = (X_smoteenn, y_smoteenn)
        
        # Random under sampling
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X, y)
        balanced_datasets['random_under'] = (X_rus, y_rus)
        
        print("Class balancing completed")
        return balanced_datasets
    
    def optimize_hyperparameters(self, X, y, model_name, model):
        """Optimize hyperparameters using Optuna."""
        print(f"Optimizing hyperparameters for {model_name}...")
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
                }
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
            else:
                return 0.0
            
            # Create model with suggested parameters
            if model_name == 'random_forest':
                clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_name == 'xgboost':
                clf = xgb.XGBClassifier(**params, random_state=42)
            elif model_name == 'lightgbm':
                clf = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            else:
                return 0.0
            
            # Cross-validation score
            cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def train_ensemble(self, X, y):
        """Train ensemble models with optimized hyperparameters."""
        print("Training ensemble models...")
        
        # Create models
        models = self.create_ensemble_models()
        
        # Train each model
        trained_models = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Optimize hyperparameters for key models
            if name in ['random_forest', 'xgboost', 'lightgbm']:
                best_params = self.optimize_hyperparameters(X, y, name, model)
                if name == 'random_forest':
                    model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
                elif name == 'xgboost':
                    model = xgb.XGBClassifier(**best_params, random_state=42)
                elif name == 'lightgbm':
                    model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
            
            # Train model
            model.fit(X, y)
            trained_models[name] = model
            
            # Evaluate
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            print(f"{name} - Accuracy: {accuracy_score(y, y_pred):.4f}, AUC: {roc_auc_score(y, y_proba):.4f}")
        
        return trained_models
    
    def create_voting_ensemble(self, trained_models, X, y):
        """Create voting ensemble from trained models."""
        print("Creating voting ensemble...")
        
        # Create voting classifier
        estimators = [(name, model) for name, model in trained_models.items()]
        voting_clf = VotingClassifier(estimators=estimators, voting='soft')
        
        # Train voting ensemble
        voting_clf.fit(X, y)
        
        return voting_clf
    
    def create_stacking_ensemble(self, trained_models, X, y):
        """Create stacking ensemble from trained models."""
        print("Creating stacking ensemble...")
        
        # Create stacking classifier
        estimators = [(name, model) for name, model in trained_models.items()]
        meta_classifier = LogisticRegression(random_state=42)
        
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_classifier,
            cv=5
        )
        
        # Train stacking ensemble
        stacking_clf.fit(X, y)
        
        return stacking_clf
    
    def evaluate_models(self, models, X, y):
        """Comprehensive model evaluation."""
        print("Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            # Predictions
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            # Metrics
            results[name] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1_score': f1_score(y, y_pred),
                'auc_roc': roc_auc_score(y, y_proba)
            }
            
            print(f"\n{name.upper()} Results:")
            print(f"Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall: {results[name]['recall']:.4f}")
            print(f"F1-Score: {results[name]['f1_score']:.4f}")
            print(f"AUC-ROC: {results[name]['auc_roc']:.4f}")
        
        return results
    
    def select_best_model(self, models, results):
        """Select the best performing model based on business metrics."""
        print("Selecting best model...")
        
        # Score each model (business-focused scoring)
        model_scores = {}
        for name, metrics in results.items():
            # Business-focused scoring: prioritize recall and AUC
            score = (metrics['recall'] * 0.4 + 
                    metrics['auc_roc'] * 0.3 + 
                    metrics['f1_score'] * 0.2 + 
                    metrics['precision'] * 0.1)
            model_scores[name] = score
        
        # Find best model
        best_model_name = max(model_scores, key=model_scores.get)
        self.best_model = models[best_model_name]
        
        print(f"Best model: {best_model_name} (Score: {model_scores[best_model_name]:.4f})")
        
        return self.best_model
    
    def save_models(self, models, best_model):
        """Save all models and best model."""
        print("Saving models...")
        
        # Save all models
        for name, model in models.items():
            joblib.dump(model, f'models/{name}_model.joblib')
        
        # Save best model
        joblib.dump(best_model, 'models/best_churn_model.joblib')
        
        # Save feature names
        joblib.dump(self.feature_names, 'models/feature_names.joblib')
        
        print("Models saved successfully")

def train_advanced_models():
    """Main function to run advanced model training."""
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Load and prepare data
    X, y = trainer.load_and_prepare_data()
    
    # Handle class imbalance
    balanced_datasets = trainer.handle_class_imbalance(X, y)
    
    # Train models on balanced datasets
    all_models = {}
    all_results = {}
    
    for balance_method, (X_balanced, y_balanced) in balanced_datasets.items():
        print(f"\n{'='*50}")
        print(f"Training with {balance_method} balanced data")
        print(f"{'='*50}")
        
        # Train ensemble models
        models = trainer.train_ensemble(X_balanced, y_balanced)
        
        # Create ensemble methods
        voting_clf = trainer.create_voting_ensemble(models, X_balanced, y_balanced)
        stacking_clf = trainer.create_stacking_ensemble(models, X_balanced, y_balanced)
        
        # Add ensemble models
        models['voting_ensemble'] = voting_clf
        models['stacking_ensemble'] = stacking_clf
        
        # Evaluate all models
        results = trainer.evaluate_models(models, X_balanced, y_balanced)
        
        # Store results
        all_models[balance_method] = models
        all_results[balance_method] = results
        
        # Select best model for this balance method
        best_model = trainer.select_best_model(models, results)
        
        # Save models for this balance method
        trainer.save_models(models, best_model)
    
    # Find overall best model across all balance methods
    overall_best_score = 0
    overall_best_model = None
    overall_best_method = None
    
    for method, results in all_results.items():
        for model_name, metrics in results.items():
            score = (metrics['recall'] * 0.4 + 
                    metrics['auc_roc'] * 0.3 + 
                    metrics['f1_score'] * 0.2 + 
                    metrics['precision'] * 0.1)
            
            if score > overall_best_score:
                overall_best_score = score
                overall_best_model = all_models[method][model_name]
                overall_best_method = f"{method}_{model_name}"
    
    print(f"\n{'='*50}")
    print(f"OVERALL BEST MODEL: {overall_best_method}")
    print(f"Score: {overall_best_score:.4f}")
    print(f"{'='*50}")
    
    # Save overall best model
    joblib.dump(overall_best_model, 'models/overall_best_churn_model.joblib')
    
    return overall_best_model, all_results

if __name__ == "__main__":
    train_advanced_models()