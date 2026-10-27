# =============================================================================
# MODEL TRAINER - training, tuning, saving and loading classifiers
# =============================================================================
import os
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import glob
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost is not installed. Install with: pip install xgboost")


class ModelML: 
    
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None):
        """Initialize the model trainer with optional pre-split data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_pred = None
        self.y_train_pred = None
        self.y_test_proba = None
        self.y_train_proba = None
        self.model = None
        self.model_name = None
        self.training_date = None
        self.best_params = None
        self.cv_results = None
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    def _validate_data(self):
        """Check that training data is available."""
        if self.x_train is None or self.y_train is None:
            raise ValueError("Training data is not defined.")
    
    def _post_training(self, model_name, params):
        """Common operations after training."""
        self.model_name = model_name
        self.training_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_params = params
        
        # Predictions on train
        self.y_train_pred = self.model.predict(self.x_train)
        if hasattr(self.model, 'predict_proba'):
            self.y_train_proba = self.model.predict_proba(self.x_train)[:, 1]
        
        # Predictions on test
        if self.x_test is not None:
            self.y_test_pred = self.model.predict(self.x_test)
            if hasattr(self.model, 'predict_proba'):
                self.y_test_proba = self.model.predict_proba(self.x_test)[:, 1]
        
        self._print_metrics()
    
    def _print_metrics(self):
        """Print quick accuracy metrics."""
        train_acc = np.mean(self.y_train_pred == self.y_train)
        print(f"\nResults:")
        print(f"   - Train Accuracy: {train_acc:.4f}")
        if self.y_test is not None and self.y_test_pred is not None:
            test_acc = np.mean(self.y_test_pred == self.y_test)
            print(f"   - Test Accuracy:  {test_acc:.4f}")
    
    def _tune_model(self, base_model, param_grid, method, cv, scoring, n_iter, model_name):
        """Generic method for hyperparameter tuning."""
        
        method_upper = method.upper()
        print(f"\nHyperparameter search ({method_upper})")
        print(f"   Model: {model_name}")
        print(f"   Scoring: {scoring}")
        print(f"   CV folds: {cv}")
        print(f"   Parameters searched: {list(param_grid.keys())}")
        
        if method_upper == 'GRID':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif method_upper == 'RANDOM':
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError("method must be 'grid' or 'random'")
        
        search.fit(self.x_train, self.y_train)
        
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_
        }
        
        print(f"\nBest parameters found:")
        for param, value in self.best_params.items():
            print(f"   - {param}: {value}")
        print(f"   - Best CV Score ({scoring}): {search.best_score_:.4f}")
        
        self._post_training(model_name, self.best_params)
    
    # =========================================================================
    # LOGISTIC REGRESSION
    # =========================================================================
    def train_logistic_regression(self, **lr_params):
        """
        Train a Logistic Regression model. 
        
        :param lr_params: Parameters for LogisticRegression
        :return: self (for method chaining)
        """
        self._validate_data()
        
        default_params = {
            'max_iter': 1000,
            'random_state': 42
        }
        default_params.update(lr_params)
        
        print(f"\nTraining Logistic Regression")
        print(f"   Parameters: {default_params}")
        
        self.model = LogisticRegression(**default_params)
        self.model.fit(self.x_train, self.y_train)
        
        self._post_training('logistic_regression', default_params)
        return self
    
    def tune_logistic_regression(self, param_grid=None, method='grid', cv=5, scoring='roc_auc', n_iter=50):
        """
        Tune Logistic Regression hyperparameters. 
        
        :param param_grid: Parameters to search (uses default if None)
        :param method: 'grid' or 'random'
        :param cv: Cross-validation folds
        :param scoring: Metric to optimize ('accuracy', 'roc_auc', 'f1', etc.)
        :param n_iter: Iterations for random search
        :return: self (for method chaining)
        """
        self._validate_data()
        
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced'],
                'max_iter': [1000]
            }
        
        self._tune_model(
            base_model=LogisticRegression(random_state=42),
            param_grid=param_grid,
            method=method,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            model_name='logistic_regression_tuned'
        )
        return self
    
    # =========================================================================
    # RANDOM FOREST
    # =========================================================================
    def train_random_forest(self, **rf_params):
        """
        Train a Random Forest model.
        
        :param rf_params: Parameters for RandomForestClassifier
        :return: self (for method chaining)
        """
        self._validate_data()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        default_params.update(rf_params)
        
        print(f"\nTraining Random Forest")
        print(f"   Parameters: {default_params}")
        
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(self.x_train, self.y_train)
        
        self._post_training('random_forest', default_params)
        return self
    
    def tune_random_forest(self, param_grid=None, method='random', cv=5, scoring='roc_auc', n_iter=50):
        """
        Tune Random Forest hyperparameters.
        
        :param param_grid: Parameters to search (uses default if None)
        :param method: 'grid' or 'random'
        :param cv: Cross-validation folds
        :param scoring: Metric to optimize
        :param n_iter: Iterations for random search
        :return: self (for method chaining)
        """
        self._validate_data()
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced']
            }
        
        self._tune_model(
            base_model=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            method=method,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            model_name='random_forest_tuned'
        )
        return self
    
    # =========================================================================
    # XGBOOST
    # =========================================================================
    def train_xgboost(self, **xgb_params):
        """
        Train an XGBoost model.
        
        :param xgb_params: Parameters for XGBClassifier
        :return: self (for method chaining)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self._validate_data()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss'
        }
        default_params.update(xgb_params)
        
        print(f"\nTraining XGBoost")
        print(f"   Parameters: {default_params}")
        
        self.model = xgb.XGBClassifier(**default_params)
        self.model.fit(self.x_train, self.y_train)
        
        self._post_training('xgboost', default_params)
        return self
    
    def tune_xgboost(self, param_grid=None, method='random', cv=5, scoring='roc_auc', n_iter=50):
        """
        Tune XGBoost hyperparameters.
        
        :param param_grid: Parameters to search (uses default if None)
        :param method: 'grid' or 'random'
        :param cv: Cross-validation folds
        :param scoring: Metric to optimize
        :param n_iter: Iterations for random search
        :return: self (for method chaining)
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
        
        self._validate_data()
        
        if param_grid is None: 
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2]
            }
        
        self._tune_model(
            base_model=xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss'),
            param_grid=param_grid,
            method=method,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            model_name='xgboost_tuned'
        )
        return self
    
    # =========================================================================
    # GRADIENT BOOSTING DECISION TREE (sklearn)
    # =========================================================================
    def train_gradient_boosting(self, **gb_params):
        """
        Train a Gradient Boosting Decision Tree model (sklearn).
        
        :param gb_params: Parameters for GradientBoostingClassifier
        :return: self (for method chaining)
        """
        self._validate_data()
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
        default_params.update(gb_params)
        
        print(f"\nTraining Gradient Boosting DT")
        print(f"   Parameters: {default_params}")
        
        self.model = GradientBoostingClassifier(**default_params)
        self.model.fit(self.x_train, self.y_train)
        
        self._post_training('gradient_boosting', default_params)
        return self
    
    def tune_gradient_boosting(self, param_grid=None, method='random', cv=5, scoring='roc_auc', n_iter=50):
        """
        Tune Gradient Boosting Decision Tree hyperparameters.
        
        :param param_grid: Parameters to search (uses default if None)
        :param method: 'grid' or 'random'
        :param cv: Cross-validation folds
        :param scoring: Metric to optimize
        :param n_iter: Iterations for random search
        :return: self (for method chaining)
        """
        self._validate_data()
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [3, 4, 5, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        self._tune_model(
            base_model=GradientBoostingClassifier(random_state=42),
            param_grid=param_grid,
            method=method,
            cv=cv,
            scoring=scoring,
            n_iter=n_iter,
            model_name='gradient_boosting_tuned'
        )
        return self
    
    # =========================================================================
    # UPDATE HYPERPARAMETERS
    # =========================================================================
    def update_hyperparameters(self, model_type=None, **new_params):
        """
        Update hyperparameters and retrain the model.
        
        :param model_type: 'logistic', 'rf', 'xgboost' (auto-detected if None)
        :param new_params: New hyperparameters to use
        :return: self (for method chaining)
        """
        if self.best_params is None:
            current_params = {}
        else:
            current_params = self.best_params.copy()
        
        # Update with new params
        current_params.update(new_params)
        
        print(f"\nUpdating hyperparameters and retraining")
        print(f"   Old: {self.best_params}")
        print(f"   New: {current_params}")
        
        # Auto-detect model type if not provided
        if model_type is None:
            if self.model_name is None:
                raise ValueError("No previous model. Specify model_type='logistic', 'rf', or 'xgboost'")
            elif 'logistic' in self.model_name.lower():
                model_type = 'logistic'
            elif 'random' in self.model_name.lower() or 'forest' in self.model_name.lower():
                model_type = 'rf'
            elif 'xgb' in self.model_name.lower() or 'xgboost' in self.model_name.lower():
                model_type = 'xgboost'
            elif 'gradient' in self.model_name.lower() or 'boosting' in self.model_name.lower():
                model_type = 'gb'
            else:
                raise ValueError(f"Unrecognized model type: {self.model_name}")
        
        # Retrain with updated params
        if model_type == 'logistic':
            return self.train_logistic_regression(**current_params)
        elif model_type == 'rf': 
            return self.train_random_forest(**current_params)
        elif model_type == 'xgboost':
            return self.train_xgboost(**current_params)
        elif model_type == 'gb':
            return self.train_gradient_boosting(**current_params)
        else:
            raise ValueError("model_type must be 'logistic', 'rf', 'xgboost', or 'gb'")
    
    # =========================================================================
    # PREDICTION METHODS
    # =========================================================================
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("No trained model.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("No trained model.")
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        raise AttributeError("This model does not support predict_proba()")
    
    # =========================================================================
    # SAVE / LOAD
    # =========================================================================
    def save_model(self, filename=None, model_dir='models', save_params=True):
        """
        Save model and hyperparameters. 
        
        :param filename: Name of the file (auto-generated if None)
        :param model_dir: Directory to save the model (default: models)
        :param save_params: If True, also save hyperparameters to JSON
        :return: Path where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        if filename is None:
            filename = f"{self.model_name}"
        
        filename = filename.replace('.pkl', '').replace('.json', '')
        
        # Save model directly in models/ directory
        model_path = os.path.join(model_dir, f"{filename}.pkl")
        joblib.dump(self.model, model_path)
        print(f"Model saved: {model_path}")
        
        # Save hyperparameters
        if save_params and self.best_params:
            params_path = os.path.join(model_dir, f"{filename}_params.json")
            params_to_save = {
                'model_name': self.model_name,
                'training_date': self.training_date,
                'hyperparameters': self.best_params,
                'cv_best_score': self.cv_results.get('best_score') if self.cv_results else None
            }
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4, default=str)
            print(f"Parameters saved: {params_path}")
        
        return model_path
    
    def load_model(self, model_path, load_params=True):
        """
        Load model and hyperparameters.
        
        :param model_path: Path to the saved model file
        :param load_params: If True, also load hyperparameters from JSON
        :return: self (for method chaining)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_name = type(self.model).__name__
        print(f"Model loaded: {model_path}")
        
        if load_params: 
            params_path = model_path.replace('.pkl', '_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params_data = json.load(f)
                self.best_params = params_data.get('hyperparameters')
                self.training_date = params_data.get('training_date')
                self.model_name = params_data.get('model_name', self.model_name)
                print(f"Parameters loaded: {self.best_params}")
        
        return self
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    def get_hyperparameters(self):
        """Return current hyperparameters."""
        if self.model is not None:
            return self.model.get_params()
        return self.best_params
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance (for RF and XGBoost).
        
        :param feature_names: List of feature names (optional)
        :return: DataFrame with features and their importance
        """
        if self.model is None:
            raise ValueError("No trained model.")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if feature_names is None:
                feature_names = [f'feat_{i}' for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        elif hasattr(self.model, 'coef_'):
            # For Logistic Regression
            coefs = np.abs(self.model.coef_[0])
            
            if feature_names is None: 
                feature_names = [f'feat_{i}' for i in range(len(coefs))]
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': coefs
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        else:
            raise AttributeError("This model does not support feature_importances_ or coef_")
    
    def summary(self):
        """Print a summary of the current model state."""
        print("\n" + "=" * 50)
        print("MODEL SUMMARY")
        print("=" * 50)
        print(f"   Model: {self.model_name}")
        print(f"   Training date: {self.training_date}")
        print(f"   Hyperparameters: {self.best_params}")
        if self.cv_results:
            print(f"   CV Best Score: {self.cv_results.get('best_score', 'N/A'):.4f}")
        if self.x_train is not None:
            print(f"   Train size: {len(self.x_train)}")
        if self.x_test is not None:
            print(f"   Test size: {len(self.x_test)}")
        print("=" * 50)

def find_existing_model(model_dir, model_type):
    """Find an existing model of the given type."""
    # Possible patterns (case-insensitive search)
    patterns = [
        os.path.join(model_dir, f"{model_type}*.pkl"),
        os.path.join(model_dir, f"{model_type.lower()}*.pkl"),
        os.path.join(model_dir, f"{model_type.replace('Regression', '_regression').lower()}*.pkl"),
    ]
    
    models = []
    for pattern in patterns:
        models.extend(glob.glob(pattern))
    
    # Exclude _params.json files
    models = [m for m in models if '_params' not in m]
    
    if models:
        # Return the most recent one
        return max(models, key=os.path.getctime)
    return None