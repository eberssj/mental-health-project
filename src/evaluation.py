import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import shap
import os
import pandas as pd

def export_metrics(y_true, y_pred, y_prob, target_names, model_name="Model", best_params=None):
    print("\n" + "="*50)
    print(f"--- {model_name.upper()} REPORT ---")
    if best_params:
        print(f"Best Parameters: {best_params}")
    print("="*50)
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
        print(f"Multi-class ROC-AUC Score: {roc_auc:.4f}")
    except Exception as e:
        print(f"Could not compute ROC-AUC: {e}")
        
    return report

def save_confusion_matrix(y_true, y_pred, target_names, filename):
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', 
                xticklabels=target_names, yticklabels=target_names, cmap='Blues')
    plt.title(f'Confusion Matrix: {filename}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    path = os.path.join('exports/plots', f'{filename}.png')
    plt.savefig(path)
    plt.close()

def save_correlation_matrix(X, filename="psychiatric_correlation"):
    """Generates and saves a psychiatric correlation heatmap for features."""
    plt.figure(figsize=(16, 14))
    # Using a sample if dataset is too large for fast plotting
    corr = X.sample(min(2000, len(X)), random_state=42).corr()
    
    # Plot only top correlated features to avoid unreadable dense charts
    top_features = corr.abs().sum().sort_values(ascending=False).head(40).index
    sns.heatmap(corr.loc[top_features, top_features], cmap='coolwarm', annot=False, center=0)
    plt.title('Psychiatric Feature Correlation Matrix (Top 40 Explanatory Features)')
    plt.tight_layout()
    
    path = os.path.join('exports/plots', f'{filename}.png')
    plt.savefig(path)
    plt.close()