import os
import torch
import numpy as np
from sklearn.metrics import brier_score_loss

from data.dataset import get_dataloaders
from models.backbones import FeatureExtractor
from models.baselines import OT_CGSF, OT_CGSF_Classifier, ResilientSinkhornFusion, DDEF
from models.proposed import OT_Evidential_Fusion
from scripts.train import train_and_evaluate
from utils.plotting import generate_performance_plots, plot_calibration_diagram
from utils.metrics import expected_calibration_error

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = '/kaggle/input'
    train_loader, val_loader = get_dataloaders(data_dir=data_dir, batch_size=8, num_samples=4000)
    
    extractor_endo = FeatureExtractor(feature_dim=512).to(device)
    extractor_histo = FeatureExtractor(feature_dim=512).to(device)
    
    extractor_endo.eval()
    extractor_histo.eval()
    for param in extractor_endo.parameters(): param.requires_grad = False
    for param in extractor_histo.parameters(): param.requires_grad = False

    models_dict = {
        "OT-CGSF (SOTA 1)": OT_CGSF_Classifier(OT_CGSF(in_channels=3)).to(device),
        "Resilient Sinkhorn (SOTA 2)": ResilientSinkhornFusion().to(device),
        "DDEF (SOTA 3)": DDEF(feature_dim=512).to(device),
        "OT-Evidential (Novel)": OT_Evidential_Fusion(feature_dim=512).to(device)
    }

    history = train_and_evaluate(models_dict, extractor_endo, extractor_histo, train_loader, val_loader, device, epochs=20)
    
    print("Generating performance plots...")
    generate_performance_plots(history, output_dir='./paper_plots')
    
    print("Computing calibration metrics for proposed model...")
    novel_history = history["OT-Evidential (Novel)"]
    y_true = np.array(novel_history['labels'])
    y_prob = np.array(novel_history['probs'])
    
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    
    print(f"Brier Score: {brier:.4f}")
    print(f"ECE: {ece:.4f}")
    
    plot_calibration_diagram(y_true, y_prob, brier, ece, output_dir='./paper_plots')
    print("Workflow complete.")

if __name__ == "__main__":
    main()