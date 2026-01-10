# =============================================================================
# CALIBRATION UTILITIES - Platt Scaling
# =============================================================================
import os
import glob
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


# Custom class for calibrated models (must be at module level for pickling)
class CalibratedModel:
    """Wrapper class for calibrated models"""
    def __init__(self, base_model, calibrator, method):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method
    
    def predict_proba(self, X):
        """Get calibrated probabilities"""
        proba = self.base_model.predict_proba(X)[:, 1]
        calibrated_proba = self.calibrator.predict(proba)
        # Stack as [proba_class_0, proba_class_1]
        return np.column_stack([1 - calibrated_proba, calibrated_proba])
    
    def predict(self, X):
        """Get calibrated predictions"""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)


def find_all_models(model_dir='models', pattern='*_tuned.pkl'):
    """
    Find all tuned models in a directory.
    
    : param model_dir: Directory containing models
    :param pattern:  Glob pattern to match model files
    :return: List of model paths
    """
    search_pattern = os.path.join(model_dir, pattern)
    models = glob.glob(search_pattern)
    
    # Exclude params files
    models = [m for m in models if '_params' not in m]
    
    print(f"📂 Modèles trouvés dans '{model_dir}':")
    for m in models:
        print(f"   - {os.path.basename(m)}")
    
    return models


def calibrate_model(model, X_cal, y_cal, method='sigmoid'):
    """
    Calibrate a pre-trained model using Platt Scaling (sigmoid) or Isotonic Regression.
    
    For pre-trained models, we fit the calibration method on X_cal without retraining the base model.
    
    :param model:  Trained sklearn model (already fitted)
    :param X_cal: Calibration features
    :param y_cal:  Calibration labels
    :param method: 'sigmoid' (Platt) or 'isotonic'
    :return: Calibrated model
    """
    print(f"   🔄 Calibration avec méthode: {method}")
    
    from sklearn.calibration import _SigmoidCalibration
    from sklearn.isotonic import IsotonicRegression
    
    # Get probabilities from pre-trained model
    proba = model.predict_proba(X_cal)[:, 1]
    
    if method == 'sigmoid':
        # Fit sigmoid calibration (Platt Scaling) on calibration data
        calibrator = _SigmoidCalibration()
        calibrator.fit(proba, y_cal)
    elif method == 'isotonic':
        # Fit isotonic regression calibration
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(proba, y_cal)
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    # Create and return calibrated model
    calibrated_model = CalibratedModel(model, calibrator, method)
    return calibrated_model


def evaluate_calibration(y_true, y_proba_uncalibrated, y_proba_calibrated, model_name='Model'):
    """
    Compare calibration metrics before and after calibration.
    
    :param y_true: True labels
    :param y_proba_uncalibrated:  Probabilities before calibration
    :param y_proba_calibrated: Probabilities after calibration
    :param model_name: Name of the model
    :return: Dict with metrics
    """
    metrics = {
        'model':  model_name,
        'brier_before': brier_score_loss(y_true, y_proba_uncalibrated),
        'brier_after': brier_score_loss(y_true, y_proba_calibrated),
        'log_loss_before': log_loss(y_true, y_proba_uncalibrated),
        'log_loss_after': log_loss(y_true, y_proba_calibrated),
        'roc_auc_before':  roc_auc_score(y_true, y_proba_uncalibrated),
        'roc_auc_after': roc_auc_score(y_true, y_proba_calibrated)
    }
    
    # Improvement
    metrics['brier_improvement'] = metrics['brier_before'] - metrics['brier_after']
    metrics['log_loss_improvement'] = metrics['log_loss_before'] - metrics['log_loss_after']
    
    return metrics


def plot_calibration_curve(y_true, y_proba_uncalibrated, y_proba_calibrated, 
                           model_name='Model', save_path=None, n_bins=10):
    """
    Plot calibration curves before and after calibration.
    
    :param y_true: True labels
    :param y_proba_uncalibrated: Probabilities before calibration
    :param y_proba_calibrated: Probabilities after calibration
    :param model_name: Name of the model
    :param save_path: Path to save the plot
    :param n_bins: Number of bins for calibration curve
    :return: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curves
    ax1 = axes[0]
    
    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], 'k--', label='Parfaitement calibré')
    
    # Before calibration
    prob_true_before, prob_pred_before = calibration_curve(
        y_true, y_proba_uncalibrated, n_bins=n_bins, strategy='uniform'
    )
    ax1.plot(prob_pred_before, prob_true_before, 's-', 
             label=f'Avant calibration (Brier={brier_score_loss(y_true, y_proba_uncalibrated):.4f})',
             color='red', markersize=8)
    
    # After calibration
    prob_true_after, prob_pred_after = calibration_curve(
        y_true, y_proba_calibrated, n_bins=n_bins, strategy='uniform'
    )
    ax1.plot(prob_pred_after, prob_true_after, 'o-', 
             label=f'Après calibration (Brier={brier_score_loss(y_true, y_proba_calibrated):.4f})',
             color='green', markersize=8)
    
    ax1.set_xlabel('Probabilité prédite moyenne')
    ax1.set_ylabel('Fraction de positifs')
    ax1.set_title(f'{model_name} - Courbe de Calibration', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # Histogram of predicted probabilities
    ax2 = axes[1]
    ax2.hist(y_proba_uncalibrated, bins=20, alpha=0.5, label='Avant calibration', color='red', edgecolor='black')
    ax2.hist(y_proba_calibrated, bins=20, alpha=0.5, label='Après calibration', color='green', edgecolor='black')
    ax2.set_xlabel('Probabilité prédite')
    ax2.set_ylabel('Fréquence')
    ax2.set_title(f'{model_name} - Distribution des Probabilités', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   ✓ Plot sauvegardé:  {save_path}")
    
    return fig


def save_calibrated_model(calibrated_model, original_model_path, output_dir='model_calibrated'):
    """
    Save calibrated model. 
    
    :param calibrated_model: Calibrated sklearn model
    :param original_model_path: Path to original model (for naming)
    :param output_dir: Output directory
    :return: Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    original_name = os.path.basename(original_model_path).replace('.pkl', '')
    calibrated_name = f"{original_name}_calibrated.pkl"
    save_path = os.path.join(output_dir, calibrated_name)
    
    joblib.dump(calibrated_model, save_path)
    print(f"   ✓ Modèle calibré sauvegardé: {save_path}")
    
    return save_path


def calibrate_all_models(
    model_dir='model',
    output_dir='model_calibrated',
    X_cal=None,
    y_cal=None,
    X_test=None,
    y_test=None,
    method='sigmoid',
    pattern='*_tuned.pkl',
    save_plots=True,
    plot_dir='model_calibrated/plots',
    save_results_dir='results'
):
    """
    Calibrate all models in a directory and evaluate them.
    
    :param model_dir: Directory containing trained models
    :param output_dir: Directory to save calibrated models
    :param X_cal:  Calibration features
    :param y_cal: Calibration labels
    :param X_test: Test features for evaluation
    :param y_test: Test labels for evaluation
    :param method: 'sigmoid' (Platt) or 'isotonic'
    :param pattern: Glob pattern to match model files
    :param save_plots: If True, save calibration plots
    :param plot_dir: Directory to save plots
    :param save_results_dir: Base directory to save detailed results (results/{model_name}_calibrated)
    :return: DataFrame with calibration results
    """
    if X_cal is None or y_cal is None:
        raise ValueError("X_cal et y_cal sont requis pour la calibration")
    
    if X_test is None or y_test is None:
        raise ValueError("X_test et y_test sont requis pour l'évaluation")
    
    # Find all models
    model_paths = find_all_models(model_dir, pattern)
    
    if len(model_paths) == 0:
        print(f"⚠️ Aucun modèle trouvé avec le pattern '{pattern}' dans '{model_dir}'")
        return None
    
    
    # Results storage
    all_results = []
    
    print(f"\n{'='*60}")
    print(f"🔄 CALIBRATION DE {len(model_paths)} MODÈLES")
    print(f"{'='*60}")
    print(f"   Méthode: {method} (Platt Scaling)" if method == 'sigmoid' else f"   Méthode: {method}")
    print(f"   Données calibration: {len(X_cal)} échantillons")
    print(f"   Données test: {len(X_test)} échantillons")
    
    for model_path in model_paths: 
        model_name = os.path.basename(model_path).replace('.pkl', '').replace('_tuned', '')
        
        print(f"\n{'─'*50}")
        print(f"📦 Modèle: {model_name}")
        print(f"{'─'*50}")
        
        # Load model
        model = joblib.load(model_path)
        print(f"   ✓ Modèle chargé: {model_path}")
        
        # Get uncalibrated probabilities on test set
        y_proba_uncalibrated = model.predict_proba(X_test)[:, 1]
        
        # Calibrate model
        calibrated_model = calibrate_model(model, X_cal, y_cal, method=method)
        
        # Get calibrated probabilities on test set
        y_proba_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = evaluate_calibration(
            y_test, 
            y_proba_uncalibrated, 
            y_proba_calibrated,
            model_name=model_name
        )
        all_results.append(metrics)
        
        # Print metrics
        print(f"\n   📊 Métriques sur Test Set:")
        print(f"      Brier Score:   {metrics['brier_before']:.4f} → {metrics['brier_after']:.4f} "
              f"({'↓' if metrics['brier_improvement'] > 0 else '↑'} {abs(metrics['brier_improvement']):.4f})")
        print(f"      Log Loss:      {metrics['log_loss_before']:.4f} → {metrics['log_loss_after']:.4f} "
              f"({'↓' if metrics['log_loss_improvement'] > 0 else '↑'} {abs(metrics['log_loss_improvement']):.4f})")
        print(f"      ROC-AUC:       {metrics['roc_auc_before']:.4f} → {metrics['roc_auc_after']:.4f}")
        
        # Create results directory for this model
        model_results_dir = os.path.join(save_results_dir, f"{model_name}_calibrated")
        os.makedirs(model_results_dir, exist_ok=True)
        
        # Plot calibration curve
        if save_plots:
            plot_path = os.path.join(model_results_dir, f"{model_name}_calibration_curve.png")
            plot_calibration_curve(
                y_test, 
                y_proba_uncalibrated, 
                y_proba_calibrated,
                model_name=model_name,
                save_path=plot_path
            )
            plt.close()
        
        # Save calibration metrics to CSV
        metrics_df = pd.DataFrame({
            'Métrique': list(metrics.keys()),
            'Valeur': list(metrics.values())
        })
        metrics_csv_path = os.path.join(model_results_dir, f'{model_name}_calibrated_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"   ✓ Métriques sauvegardées: {metrics_csv_path}")
        
        # Save calibration results to TXT
        results_txt_path = os.path.join(model_results_dir, f'{model_name}_calibrated_results.txt')
        with open(results_txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"RÉSULTATS CALIBRATION - {model_name}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Méthode: {method} (Platt Scaling)\n")
            f.write(f"Données calibration: {len(X_cal)} échantillons\n")
            f.write(f"Données test: {len(X_test)} échantillons\n\n")
            
            f.write(f"{'─'*60}\n")
            f.write("COMPARAISON AVANT/APRÈS CALIBRATION\n")
            f.write(f"{'─'*60}\n\n")
            
            f.write(f"Brier Score:\n")
            f.write(f"   Avant:  {metrics['brier_before']:.6f}\n")
            f.write(f"   Après:  {metrics['brier_after']:.6f}\n")
            f.write(f"   Amélioration: {metrics['brier_improvement']:.6f}\n\n")
            
            f.write(f"Log Loss:\n")
            f.write(f"   Avant:  {metrics['log_loss_before']:.6f}\n")
            f.write(f"   Après:  {metrics['log_loss_after']:.6f}\n")
            f.write(f"   Amélioration: {metrics['log_loss_improvement']:.6f}\n\n")
            
            f.write(f"ROC-AUC:\n")
            f.write(f"   Avant:  {metrics['roc_auc_before']:.6f}\n")
            f.write(f"   Après:  {metrics['roc_auc_after']:.6f}\n")
            f.write(f"   Changement: {metrics['roc_auc_after'] - metrics['roc_auc_before']:+.6f}\n\n")
        
        print(f"   ✓ Résultats sauvegardés: {results_txt_path}")
        
        # Save calibrated model
        save_calibrated_model(calibrated_model, model_path, output_dir)
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save overall results to CSV
    results_path = os.path.join(output_dir, 'calibration_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Résultats globaux sauvegardés: {results_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("📋 RÉSUMÉ DE LA CALIBRATION")
    print(f"{'='*60}")
    print(results_df[['model', 'brier_before', 'brier_after', 'brier_improvement']].to_string(index=False))
    
    return results_df


def plot_all_calibration_comparison(results_df, save_path=None):
    """
    Plot comparison of calibration improvement across all models.
    
    : param results_df: DataFrame from calibrate_all_models
    :param save_path: Path to save the plot
    :return: Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x = np.arange(len(results_df))
    width = 0.35
    
    # Brier Score comparison
    ax1 = axes[0]
    bars1 = ax1.bar(x - width/2, results_df['brier_before'], width, label='Avant', color='red', alpha=0.7)
    bars2 = ax1.bar(x + width/2, results_df['brier_after'], width, label='Après', color='green', alpha=0.7)
    ax1.set_xlabel('Modèle')
    ax1.set_ylabel('Brier Score (↓ meilleur)')
    ax1.set_title('Comparaison Brier Score', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3, axis='y')
    
    # Log Loss comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width/2, results_df['log_loss_before'], width, label='Avant', color='red', alpha=0.7)
    bars4 = ax2.bar(x + width/2, results_df['log_loss_after'], width, label='Après', color='green', alpha=0.7)
    ax2.set_xlabel('Modèle')
    ax2.set_ylabel('Log Loss (↓ meilleur)')
    ax2.set_title('Comparaison Log Loss', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['model'], rotation=45, ha='right')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Comparaison sauvegardée:  {save_path}")
    
    return fig