import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any
import warnings
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, mean_squared_error, brier_score_loss

# ========================================================================
# Model class
# ========================================================================

class model_ml():
    
    def __init__(self,x , y ):
        self.x = x 
        self.y = y 
        self.x_test = None
        self.y_test = None 
        self.x_train = None
        self.y_train = None 
        self.y_test_pred = None
        self.y_train_pred = None 
        self.model = None
        
    def split_set(self):
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x, self.y, test_size=0.2 , random_state = 42)
        
    def metrics(self,set : str): 
        if set is "train" : 
            return {
                'Accuracy': accuracy_score(self.y_train, self.y_train_pred),
                'Precision': precision_score(self.y_train, self.y_train_pred, average='weighted', zero_division=0),
                'Recall': recall_score(self.y_train, self.y_train_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(self.y_train, self.y_train_pred, average='weighted', zero_division=0),
          }
        elif set is "test" : 
            return {
                'Accuracy': accuracy_score(self.y_test, self.y_test_pred),
                'Precision': precision_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0),
                'Recall': recall_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0),
                'F1-Score': f1_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0),
          }
        else : 
            raise ValueError("Choissisez Train ou Test dans la méthode ")
        
    def logistic_regression(self, standardize=True, max_iter=1000, random_state=42):
        """
        Effectue une régression logistique sur les données.
        
        Parameters:
        -----------
        standardize : bool, default=True
            Si True, standardise les features avant l'entraînement
        max_iter : int, default=1000
            Nombre maximum d'itérations pour la convergence
        random_state : int, default=42
            Graine aléatoire pour la reproductibilité
        
        Returns:
        --------
        dict : Dictionnaire contenant le modèle et les métriques
        """
        
        # Vérifier que les données sont divisées
        if self.x_train is None or self.x_test is None:
            print("Division des données en cours...")
            self.split_set()
        
        # Vérifier si y est binaire, sinon le convertir
        unique_values = np.unique(self.y)
        if len(unique_values) > 2:
            # Si c'est continu, créer une variable binaire (médiane comme seuil)
            median_threshold = np.median(self.y)
            y_binary = (self.y > median_threshold).astype(int)
            y_train_binary = (self.y_train > median_threshold).astype(int)
            y_test_binary = (self.y_test > median_threshold).astype(int)
            print(f"Variable cible convertie en binaire avec seuil médian: {median_threshold:.3f}")
            print(f"Distribution des classes: {np.bincount(y_binary)}")
        else:
            y_train_binary = self.y_train.astype(int)
            y_test_binary = self.y_test.astype(int)
        
        # Standardisation des features
        if standardize:
            self.scaler = StandardScaler()
            x_train_processed = self.scaler.fit_transform(self.x_train)
            x_test_processed = self.scaler.transform(self.x_test)
        else:
            x_train_processed = self.x_train
            x_test_processed = self.x_test
        
        # Entraînement du modèle
        self.logistic_model = LogisticRegression(
            random_state=random_state, 
            max_iter=max_iter
        )
        self.logistic_model.fit(x_train_processed, y_train_binary)
        
        # Prédictions
        self.y_train_pred = self.logistic_model.predict(x_train_processed)
        self.y_test_pred = self.logistic_model.predict(x_test_processed)
        
        # Probabilités pour ROC-AUC
        y_train_proba = self.logistic_model.predict_proba(x_train_processed)[:, 1]
        y_test_proba = self.logistic_model.predict_proba(x_test_processed)[:, 1]
        
        # Métriques avec ROC-AUC
        train_metrics = {
            'Accuracy': accuracy_score(y_train_binary, self.y_train_pred),
            'Precision': precision_score(y_train_binary, self.y_train_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_train_binary, self.y_train_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_train_binary, self.y_train_pred, average='weighted', zero_division=0),
            'ROC-AUC': roc_auc_score(y_train_binary, y_train_proba) if len(np.unique(y_train_binary)) > 1 else 0
        }
        
        test_metrics = {
            'Accuracy': accuracy_score(y_test_binary, self.y_test_pred),
            'Precision': precision_score(y_test_binary, self.y_test_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test_binary, self.y_test_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test_binary, self.y_test_pred, average='weighted', zero_division=0),
            'ROC-AUC': roc_auc_score(y_test_binary, y_test_proba) if len(np.unique(y_test_binary)) > 1 else 0
        }
        
        # Tableau de comparaison
        comparison_df = pd.DataFrame({
            'Métrique': list(train_metrics.keys()),
            'Train': [f"{v:.4f}" for v in train_metrics.values()],
            'Test': [f"{v:.4f}" for v in test_metrics.values()]
        })
        
        # Différence pour détecter l'overfitting
        comparison_df['Différence (Train-Test)'] = [
            f"{train_metrics[metric] - test_metrics[metric]:.4f}" 
            for metric in train_metrics.keys()
        ]
        
        print("=== RÉSULTATS DE LA RÉGRESSION LOGISTIQUE ===")
        print(f"Taille du dataset: {len(self.x)} échantillons")
        print(f"Nombre de features: {self.x.shape[1]}")
        print(f"Train set: {len(self.x_train)} échantillons")
        print(f"Test set: {len(self.x_test)} échantillons")
        print("\n=== TABLEAU DE COMPARAISON ===")
        print(comparison_df.to_string(index=False))
        
        # Matrices de confusion
        print("\n=== MATRICES DE CONFUSION ===")
        print("Train:")
        print(confusion_matrix(y_train_binary, self.y_train_pred))
        print("\nTest:")
        print(confusion_matrix(y_test_binary, self.y_test_pred))
        
        # Importance des features (coefficients)
        if hasattr(self.x, 'columns'):
            feature_names = self.x.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(self.x.shape[1])]
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': self.logistic_model.coef_[0],
            'Abs_Coefficient': np.abs(self.logistic_model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\n=== IMPORTANCE DES FEATURES (Top 10) ===")
        print(feature_importance.head(10).to_string(index=False))
        
        # Rapport de classification détaillé
        print("\n=== RAPPORT DE CLASSIFICATION (Test Set) ===")
        print(classification_report(y_test_binary, self.y_test_pred))
        
        return {
            'model': self.logistic_model,
            'scaler': self.scaler if standardize else None,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'comparison_df': comparison_df,
            'feature_importance': feature_importance,
            'y_train_binary': y_train_binary,
            'y_test_binary': y_test_binary,
            'y_train_proba': y_train_proba,
            'y_test_proba': y_test_proba
        }

    def plot_logistic_results(self):
        """
        Affiche des graphiques pour analyser les résultats de la régression logistique.
        """
        if not hasattr(self, 'logistic_model'):
            print("Aucun modèle logistique trouvé. Exécutez d'abord logistic_regression().")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Matrice de confusion
        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_predictions(
            self.y_test if hasattr(self, 'y_test_binary') else self.y_test, 
            self.y_test_pred, 
            ax=axes[0,0]
        )
        axes[0,0].set_title('Matrice de Confusion (Test Set)')
        
        # 2. Distribution des probabilités
        y_test_proba = self.logistic_model.predict_proba(
            self.scaler.transform(self.x_test) if hasattr(self, 'scaler') and self.scaler else self.x_test
        )[:, 1]
        
        axes[0,1].hist(y_test_proba, bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Distribution des Probabilités Prédites')
        axes[0,1].set_xlabel('Probabilité')
        axes[0,1].set_ylabel('Fréquence')
        
        # 3. Importance des features (top 10)
        if hasattr(self.x, 'columns'):
            feature_names = self.x.columns
        else:
            feature_names = [f'Feature_{i}' for i in range(self.x.shape[1])]
        
        coefficients = self.logistic_model.coef_[0]
        top_indices = np.argsort(np.abs(coefficients))[-10:]
        
        axes[1,0].barh(range(10), coefficients[top_indices])
        axes[1,0].set_yticks(range(10))
        axes[1,0].set_yticklabels([feature_names[i] for i in top_indices])
        axes[1,0].set_title('Top 10 Coefficients (Importance des Features)')
        axes[1,0].set_xlabel('Coefficient')
        
        # 4. Courbe ROC
        from sklearn.metrics import roc_curve, auc
        y_test_binary = getattr(self, 'y_test_binary', self.y_test)
        fpr, tpr, _ = roc_curve(y_test_binary, y_test_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1,1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1,1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1,1].set_xlim([0.0, 1.0])
        axes[1,1].set_ylim([0.0, 1.05])
        axes[1,1].set_xlabel('Taux de Faux Positifs')
        axes[1,1].set_ylabel('Taux de Vrais Positifs')
        axes[1,1].set_title('Courbe ROC')
        axes[1,1].legend(loc="lower right")
        
        plt.tight_layout()
        plt.show()

    def isotonic_regression(self, x: Optional[NDArray[np.float64]] = None, y: Optional[NDArray[np.float64]] = None, model: str = "logistique", plot: bool = True):
        """
        Predicts probabilities using the chosen base model, calibrates them using Isotonic Regression,
        and compares the RMSE of the uncalibrated vs. calibrated probabilities.
        
        Data Split: 70% Train (Base), 20% Calibrate (Isotonic), 10% Test (Final Eval).
        """
        from sklearn.metrics import mean_squared_error, brier_score_loss

        # 1. Preparation of Data
        X_data = x if x is not None else self.x
        Y_data = y if y is not None else self.y
        
        # Handle continuous Y if necessary
        if len(np.unique(Y_data)) > 2:
            median_threshold = np.median(Y_data)
            Y_binary = (Y_data > median_threshold).astype(int)
            print(f"Isotonic: Variable cible convertie en binaire (Seuil: {median_threshold:.3f})")
        else:
            Y_binary = Y_data.astype(int)

        # 2. Splitting the Data (70% Train, 20% Calibration, 10% Test)
        # Step A: Split into (90% Temp) and (10% Final Test)
        X_temp, X_test_final, y_temp, y_test_final = train_test_split(
            X_data, Y_binary, test_size=0.10, random_state=42, stratify=Y_binary
        )
        
        # Step B: Split the 90% Temp into Train (70% total) and Calib (20% total)
        # 0.20 (total) / 0.90 (temp) = approx 0.222
        calib_size = 2/9
        X_train_base, X_calib, y_train_base, y_calib = train_test_split(
            X_temp, y_temp, test_size=calib_size, random_state=42, stratify=y_temp
        )

        print("\n=== SPLIT DES DONNÉES (ISOTONIC) ===")
        print(f"1. Train Base Model (70%): {len(X_train_base)}")
        print(f"2. Calibration Set (20%):  {len(X_calib)}")
        print(f"3. Final Test Set (10%):   {len(X_test_final)}")

        # 3. Train Base Model (Logistic)
        scaler_iso = StandardScaler()
        X_train_base_sc = scaler_iso.fit_transform(X_train_base)
        X_calib_sc = scaler_iso.transform(X_calib)
        X_test_final_sc = scaler_iso.transform(X_test_final)

        base_clf = None
        if model == "logistique":
            base_clf = LogisticRegression(max_iter=1000, random_state=42)
            base_clf.fit(X_train_base_sc, y_train_base)
        else:
            raise ValueError(f"Modèle '{model}' non supporté pour le moment.")

        # 4. Get Uncalibrated Probabilities
        probs_calib_uncalibrated = base_clf.predict_proba(X_calib_sc)[:, 1]
        probs_test_uncalibrated = base_clf.predict_proba(X_test_final_sc)[:, 1]

        # 5. Fit Isotonic Regression
        iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
        iso_reg.fit(probs_calib_uncalibrated, y_calib)

        # 6. Transform Test Probabilities (Calibrate them)
        probs_test_calibrated = iso_reg.transform(probs_test_uncalibrated)

        # 7. Evaluation Metrics (Calculating RMSE)
        # RMSE = Sqrt(Mean Squared Error)
        
        # Base (Logistic Only)
        mse_base = mean_squared_error(y_test_final, probs_test_uncalibrated)
        rmse_base = np.sqrt(mse_base)
        
        # Isotonic (Logistic + Calibration)
        mse_iso = mean_squared_error(y_test_final, probs_test_calibrated)
        rmse_iso = np.sqrt(mse_iso)

        # Display Results Table
        res_df = pd.DataFrame({
            'Metric': ['RMSE (Lower is better)', 'MSE (Brier Score)'],
            'Logistic (Uncalibrated)': [f"{rmse_base:.5f}", f"{mse_base:.5f}"],
            'Isotonic (Calibrated)':   [f"{rmse_iso:.5f}", f"{mse_iso:.5f}"],
            'Improvement': [f"{(rmse_base - rmse_iso):.5f}", f"{(mse_base - mse_iso):.5f}"]
        })

        print("\n=== COMPARAISON DES RÉSULTATS (TEST SET) ===")
        print(res_df.to_string(index=False))

        # 8. Plotting (Reliability Diagram)
        if plot:
            from sklearn.calibration import calibration_curve
            plt.figure(figsize=(10, 6))
            
            # Base Model Curve
            prob_true_base, prob_pred_base = calibration_curve(y_test_final, probs_test_uncalibrated, n_bins=10)
            plt.plot(prob_pred_base, prob_true_base, marker='o', linewidth=1, label=f'Logistic (RMSE={rmse_base:.4f})', color='red')
            
            # Isotonic Curve
            prob_true_iso, prob_pred_iso = calibration_curve(y_test_final, probs_test_calibrated, n_bins=10)
            plt.plot(prob_pred_iso, prob_true_iso, marker='s', linewidth=1, label=f'Isotonic (RMSE={rmse_iso:.4f})', color='blue')
            
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
            
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title('Reliability Diagram: Logistic vs Isotonic')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        return {
            'metrics': res_df,
            'rmse_base': rmse_base,
            'rmse_iso': rmse_iso,
            'probs_uncalibrated': probs_test_uncalibrated,
            'probs_calibrated': probs_test_calibrated,
            'y_test': y_test_final
        }
