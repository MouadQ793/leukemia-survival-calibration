# =============================================================================
# EVALUATION UTILITIES
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
    log_loss,
    matthews_corrcoef,
    balanced_accuracy_score,
    cohen_kappa_score
)


def compute_all_metrics(y_true, y_pred, y_proba):
    """
    Compute all classification and probabilistic metrics.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_proba: Predicted probabilities
    :return: Dict of metrics
    """
    metrics = {
        # Classification metrics
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall_Sensitivity': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'F1_Score': f1_score(y_true, y_pred, zero_division=0),
        'Matthews_Corr_Coef': matthews_corrcoef(y_true, y_pred),
        'Cohen_Kappa': cohen_kappa_score(y_true, y_pred),
        
        # Probabilistic metrics
        'ROC_AUC': roc_auc_score(y_true, y_proba),
        'PR_AUC': average_precision_score(y_true, y_proba),
        'Brier_Score': brier_score_loss(y_true, y_proba),
        'Log_Loss': log_loss(y_true, y_proba)
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print metrics in a formatted way.
    
    :param metrics: Dict of metrics
    """
    print("=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    classification_metrics = ['Accuracy', 'Balanced_Accuracy', 'Precision', 
                              'Recall_Sensitivity', 'Specificity', 'F1_Score',
                              'Matthews_Corr_Coef', 'Cohen_Kappa']
    
    probabilistic_metrics = ['ROC_AUC', 'PR_AUC', 'Brier_Score', 'Log_Loss']
    
    print("\nClassification metrics:")
    print("-" * 40)
    for name in classification_metrics:
        if name in metrics:
            print(f"   {name:25s}: {metrics[name]:.4f}")
    
    print("\nProbabilistic metrics:")
    print("-" * 40)
    for name in probabilistic_metrics:
        if name in metrics:
            print(f"   {name:25s}: {metrics[name]:.4f}")


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    """
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))


def print_confusion_matrix_details(y_true, y_pred):
    """
    Print confusion matrix details.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :return: Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    print(f"\n   True Negatives (TN):  {tn}")
    print(f"   False Positives (FP): {fp}")
    print(f"   False Negatives (FN): {fn}")
    print(f"   True Positives (TP):  {tp}")
    
    return cm


def plot_confusion_matrix(y_true, y_pred, save_path=None, ax=None):
    """
    Plot confusion matrix.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param save_path: Path to save the plot
    :param ax: Matplotlib axis (optional)
    :return: Matplotlib axis
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    ax.set_title('Confusion matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return ax


def plot_roc_curve(y_true, y_proba, save_path=None, ax=None):
    """
    Plot ROC curve.
    
    :param y_true: True labels
    :param y_proba: Predicted probabilities
    :param save_path: Path to save the plot
    :param ax: Matplotlib axis (optional)
    :return: Matplotlib axis
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.2)
    ax.set_xlabel('False positive rate (1 - specificity)')
    ax.set_ylabel('True positive rate (sensitivity)')
    ax.set_title('ROC curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return ax


def plot_precision_recall_curve(y_true, y_proba, save_path=None, ax=None):
    """
    Plot Precision-Recall curve.
    
    :param y_true: True labels
    :param y_proba: Predicted probabilities
    :param save_path: Path to save the plot
    :param ax: Matplotlib axis (optional)
    :return: Matplotlib axis
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    baseline = y_true.mean() if hasattr(y_true, 'mean') else np.mean(y_true)
    
    if ax is None: 
        fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUC = {pr_auc:.3f})')
    ax.fill_between(recall, precision, alpha=0.2, color='green')
    ax.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return ax



def plot_probability_distribution(y_true, y_proba, save_path=None, ax=None):
    """
    Plot probability distribution by class.
    
    :param y_true: True labels
    :param y_proba: Predicted probabilities
    :param save_path: Path to save the plot
    :param ax: Matplotlib axis (optional)
    :return: Matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    y_true_array = np.array(y_true)
    
    ax.hist(y_proba[y_true_array == 0], bins=20, alpha=0.6, 
            label='Class 0', color='blue', edgecolor='black')
    ax.hist(y_proba[y_true_array == 1], bins=20, alpha=0.6, 
            label='Class 1', color='red', edgecolor='black')
    ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold 0.5')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Probability distribution by class', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
    
    return ax


def plot_all_evaluation(y_true, y_pred, y_proba, save_dir=None, model_name='model'):
    """
    Generate all evaluation plots in a single figure.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_proba: Predicted probabilities
    :param save_dir: Directory to save plots
    :param model_name: Name of the model for file naming
    :return: Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot all
    plot_confusion_matrix(y_true, y_pred, ax=axes[0, 0])
    plot_roc_curve(y_true, y_proba, ax=axes[0, 1])
    plot_precision_recall_curve(y_true, y_proba, ax=axes[1, 0])
    plot_probability_distribution(y_true, y_proba, ax=axes[1, 1])
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{model_name}_evaluation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Figure saved: {save_path}")
    
    return fig


def save_metrics_to_csv(metrics, save_dir, model_name='model'):
    """
    Save metrics to CSV file.
    
    :param metrics: Dict of metrics
    :param save_dir: Directory to save
    :param model_name: Name of the model
    :return: Path to saved file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    
    save_path = os.path.join(save_dir, f'{model_name}_metrics.csv')
    metrics_df.to_csv(save_path, index=False)
    print(f"✓ Metrics saved: {save_path}")
    
    return save_path


def full_evaluation(y_true, y_pred, y_proba, model_name='model', save_dir='results'):
    """
    Run full evaluation: compute metrics, print reports, generate plots, save results.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_proba: Predicted probabilities
    :param model_name: Name of the model
    :param save_dir: Base directory to save results (default: results)
    :return: Dict of metrics
    """
    # Create save directory for this model in results/model_name
    model_save_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"FULL EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    # 1. Compute metrics
    metrics = compute_all_metrics(y_true, y_pred, y_proba)
    
    # 2. Print metrics
    print_metrics(metrics)
    
    # 3. Print classification report
    print_classification_report(y_true, y_pred)
    
    # 4. Print confusion matrix details
    print_confusion_matrix_details(y_true, y_pred)
    
    # 5. Generate and save plots
    print(f"\nGenerating plots...")
    plot_all_evaluation(y_true, y_pred, y_proba, 
                        save_dir=model_save_dir, 
                        model_name=model_name)
    
    # 6. Save individual plots
    plot_confusion_matrix(y_true, y_pred, 
                          save_path=os.path.join(model_save_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    plot_roc_curve(y_true, y_proba, 
                   save_path=os.path.join(model_save_dir, f'{model_name}_roc_curve.png'))
    plt.close()
    
    plot_precision_recall_curve(y_true, y_proba, 
                                save_path=os.path.join(model_save_dir, f'{model_name}_pr_curve.png'))
    plt.close()
    
    plot_probability_distribution(y_true, y_proba, 
                                  save_path=os.path.join(model_save_dir, f'{model_name}_prob_dist.png'))
    plt.close()
    
    # 7. Save metrics to CSV
    save_metrics_to_csv(metrics, model_save_dir, model_name)
    
    # 8. Save complete results to a text file
    results_path = os.path.join(model_save_dir, f'{model_name}_results.txt')
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write(f"FULL EVALUATION: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        
        # Classification metrics
        f.write("CLASSIFICATION METRICS\n")
        f.write("-" * 40 + "\n")
        classification_metrics = ['Accuracy', 'Balanced_Accuracy', 'Precision', 
                                  'Recall_Sensitivity', 'Specificity', 'F1_Score',
                                  'Matthews_Corr_Coef', 'Cohen_Kappa']
        for name in classification_metrics:
            if name in metrics:
                f.write(f"   {name:25s}: {metrics[name]:.4f}\n")
        
        # Probabilistic metrics
        f.write("\nPROBABILISTIC METRICS\n")
        f.write("-" * 40 + "\n")
        probabilistic_metrics = ['ROC_AUC', 'PR_AUC', 'Brier_Score', 'Log_Loss']
        for name in probabilistic_metrics:
            if name in metrics:
                f.write(f"   {name:25s}: {metrics[name]:.4f}\n")
        
        # Confusion matrix
        f.write("\nCONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"   True Negatives (TN):  {tn}\n")
        f.write(f"   False Positives (FP): {fp}\n")
        f.write(f"   False Negatives (FN): {fn}\n")
        f.write(f"   True Positives (TP):  {tp}\n")
        
        # Classification report
        f.write("\nCLASSIFICATION REPORT\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_true, y_pred, target_names=['Class 0', 'Class 1']))
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"✓ Results saved to: {results_path}")
    
    print(f"\nEvaluation complete. Results saved to: {model_save_dir}")
    
    return metrics