import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.5,
    'lines.linewidth': 2.5,
    'figure.autolayout': True
})

def clean_name(name):
    return name.replace(' ', '_').replace('(', '').replace(')', '')

def generate_performance_plots(history, output_dir='./paper_plots'):
    os.makedirs(output_dir, exist_ok=True)
    colors = ['gray', 'crimson', 'royalblue', 'teal']

    for name, hist in history.items():
        safe_name = clean_name(name)
        
        # Loss Curve
        plt.figure(figsize=(8, 6))
        plt.plot(hist['train_loss'], label='Train Loss', color='teal')
        plt.plot(hist['val_loss'], label='Validation Loss', color='darkorange', linestyle='--')
        plt.title(f"Convergence Loss: {name}", fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.savefig(os.path.join(output_dir, f'{safe_name}_Loss_Curve.png'))
        plt.close()
        
        # Confusion Matrix
        cm = confusion_matrix(hist['labels'], hist['preds'])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'], annot_kws={"size": 16})
        plt.title(f"Confusion Matrix: {name}", fontweight='bold')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(os.path.join(output_dir, f'{safe_name}_Confusion_Matrix.png'))
        plt.close()

    # ROC Curve Comparison
    plt.figure(figsize=(8, 8))
    for idx, (name, hist) in enumerate(history.items()):
        fpr, tpr, _ = roc_curve(hist['labels'], hist['probs'])
        plt.plot(fpr, tpr, color=colors[idx % len(colors)], label=f"{name.split(' ')[0]} (AUC={auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontweight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'Comparison_ROC_Curve.png'))
    plt.close()

def plot_calibration_diagram(y_true, y_prob, brier, ece, output_dir='./paper_plots'):
    os.makedirs(output_dir, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
    
    fig, ax1 = plt.subplots(figsize=(8, 6.5))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    label_text = f"OT-Evidential (Ours)\nBrier Score: {brier:.4f}\nECE: {ece:.4f}"
    ax1.plot(prob_pred, prob_true, marker='D', markersize=8, color='#008837', label=label_text)
    
    ax1.set_xlabel("Mean Predicted Probability (Confidence)", fontweight='bold')
    ax1.set_ylabel("Fraction of Positives (Accuracy)", fontweight='bold')
    ax1.set_title("Reliability Diagram and Model Sharpness", fontweight='bold', pad=15)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc="center right", frameon=True, edgecolor='black', facecolor='white', framealpha=1.0)
    
    ax2 = ax1.twinx()
    ax2.hist(y_prob, bins=40, color='#008837', alpha=0.15, edgecolor='#008837', linewidth=1, log=True)
    ax2.set_ylabel("Prediction Count (Log Scale Density)", color='#008837', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#008837')
    ax2.set_ylim([0.5, len(y_prob)]) 
    
    fig.savefig(os.path.join(output_dir, "Fig_Calibration_OT_Evidential.png"), bbox_inches='tight')
    plt.close(fig)