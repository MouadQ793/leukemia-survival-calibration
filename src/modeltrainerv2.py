import os
import joblib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from datetime import datetime

class ModelML():
    
    def __init__(self, x_train=None, x_test=None, y_train=None, y_test=None):
        """Initialize the model trainer with optional pre-split data."""
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.y_test_pred = None
        self.y_train_pred = None 
        self.model = None
        self. model_name = None
        self.training_date = None
        self.best_params = None
        self.cv_results = None
    
    def train_logistic_regression(self, **lr_params):
        """
        Train a Logistic Regression model with custom hyperparameters.
        
        :param lr_params:  Parameters to pass to LogisticRegression
        :return: self (for method chaining)
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("Les données d'entraînement ne sont pas définies.")
        
        # Default parameters
        default_params = {
            'max_iter': 1000,
            'random_state':  42
        }
        default_params.update(lr_params)
        
        print(f"\n🔄 Entraînement de Logistic Regression")
        print(f"   Paramètres: {default_params}")
        
        self.model = LogisticRegression(**default_params)
        self.model.fit(self.x_train, self.y_train)
        
        self.model_name = 'LogisticRegression'
        self.training_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_params = default_params
        
        # Predictions
        self.y_train_pred = self.model.predict(self.x_train)
        if self.x_test is not None:
            self.y_test_pred = self.model.predict(self. x_test)
        
        self._print_quick_metrics()
        
        return self
    
    def tune_logistic_regression(self, param_grid=None, method='grid', cv=5, scoring='accuracy', n_iter=50):
        """
        Tune Logistic Regression hyperparameters using GridSearch or RandomizedSearch.
        
        :param param_grid: Dict of parameters to search (uses default if None)
        :param method: 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        :param cv: Number of cross-validation folds
        :param scoring:  Metric to optimize ('accuracy', 'roc_auc', 'f1', etc.)
        :param n_iter: Number of iterations for RandomizedSearch
        :return: self (for method chaining)
        """
        if self.x_train is None or self.y_train is None:
            raise ValueError("Les données d'entraînement ne sont pas définies.")
        
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'class_weight': [None, 'balanced'],
                'max_iter': [1000]
            }
        
        print(f"\n🔍 Recherche d'hyperparamètres ({method. upper()})")
        print(f"   Scoring: {scoring}")
        print(f"   CV folds: {cv}")
        print(f"   Paramètres testés: {list(param_grid.keys())}")
        
        base_model = LogisticRegression(random_state=42)
        
        if method == 'grid':
            search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=cv, 
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif method == 'random':
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
            raise ValueError("method doit être 'grid' ou 'random'")
        
        search.fit(self.x_train, self.y_train)
        
        # Store results
        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'cv_results': search.cv_results_
        }
        self. model_name = 'LogisticRegression_tuned'
        self.training_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Predictions
        self.y_train_pred = self.model. predict(self.x_train)
        if self.x_test is not None:
            self. y_test_pred = self. model.predict(self.x_test)
        
        print(f"\n✓ Meilleurs paramètres trouvés:")
        for param, value in self.best_params.items():
            print(f"   - {param}: {value}")
        print(f"   - Best CV Score ({scoring}): {search.best_score_:.4f}")
        
        self._print_quick_metrics()
        
        return self
    
    def update_hyperparameters(self, **new_params):
        """
        Update hyperparameters and retrain the model.
        
        :param new_params: New hyperparameters to use
        :return:  self (for method chaining)
        """
        if self.best_params is None:
            current_params = {'max_iter': 1000, 'random_state': 42}
        else:
            current_params = self.best_params. copy()
        
        # Update with new params
        current_params.update(new_params)
        
        print(f"\n🔄 Mise à jour des hyperparamètres et ré-entraînement")
        print(f"   Anciens:  {self.best_params}")
        print(f"   Nouveaux:  {current_params}")
        
        return self.train_logistic_regression(**current_params)
    
    def save_model(self, filename=None, model_dir='model', save_params=True):
        """
        Save the trained model and optionally its hyperparameters.
        
        :param filename: Name of the file (auto-generated if None)
        :param model_dir: Directory to save the model
        :param save_params: If True, also save hyperparameters to JSON
        :return: Path where model was saved
        """
        if self.model is None:
            raise ValueError("Aucun modèle entraîné à sauvegarder.")
        
        os.makedirs(model_dir, exist_ok=True)
        
        if filename is None:
            filename = f"{self.model_name}_{self.training_date}"
        
        # Remove extension if provided
        filename = filename.replace('.pkl', '').replace('.json', '')
        
        # Save model
        model_path = os.path.join(model_dir, f"{filename}.pkl")
        joblib.dump(self.model, model_path)
        print(f"✓ Modèle sauvegardé:  {model_path}")
        
        # Save hyperparameters
        if save_params and self.best_params:
            params_path = os.path.join(model_dir, f"{filename}_params.json")
            params_to_save = {
                'model_name': self.model_name,
                'training_date': self.training_date,
                'hyperparameters': self.best_params,
                'cv_best_score': self.cv_results. get('best_score') if self.cv_results else None
            }
            with open(params_path, 'w') as f:
                json.dump(params_to_save, f, indent=4)
            print(f"✓ Paramètres sauvegardés:  {params_path}")
        
        return model_path
    
    def load_model(self, model_path, load_params=True):
        """
        Load a previously saved model and its hyperparameters.
        
        :param model_path:  Path to the saved model file
        :param load_params: If True, also load hyperparameters from JSON
        :return: self (for method chaining)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_name = type(self.model).__name__
        print(f"✓ Modèle chargé: {model_path}")
        
        # Try to load params
        if load_params: 
            params_path = model_path.replace('.pkl', '_params.json')
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    params_data = json.load(f)
                self.best_params = params_data.get('hyperparameters')
                self.training_date = params_data.get('training_date')
                print(f"✓ Paramètres chargés: {params_path}")
                print(f"   Hyperparamètres: {self.best_params}")
        
        return self
    
    def get_hyperparameters(self):
        """Return current hyperparameters."""
        if self.model is not None:
            return self.model.get_params()
        return self.best_params
    
    def _print_quick_metrics(self):
        """Print quick accuracy metrics."""
        train_acc = np.mean(self.y_train_pred == self.y_train)
        print(f"\n📊 Résultats:")
        print(f"   - Train Accuracy: {train_acc:.4f}")
        if self.y_test is not None and self.y_test_pred is not None:
            test_acc = np.mean(self.y_test_pred == self.y_test)
            print(f"   - Test Accuracy: {test_acc:.4f}")