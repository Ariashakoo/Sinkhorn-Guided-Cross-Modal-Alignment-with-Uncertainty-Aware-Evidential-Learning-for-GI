# Sinkhorn-Guided-Cross-Modal-Alignment-with-Uncertainty-Aware-Evidential-Learning-for-GI
Implementation of the paper "Sinkhorn Guided Cross Modal Alignment with Uncertainty Aware Evidential Learning for Gastrointestinal Cancer Detection"

This repository contains the PyTorch implementation of a novel multimodal fusion architecture designed for medical image classification. The framework combines the spatial alignment capabilities of Sinkhorn Optimal Transport with the uncertainty-awareness of Subjective Logic (Evidential Deep Learning) to fuse Endoscopy and Histopathology imagery.

## Project Overview

In medical diagnostics, combining macroscopic findings (e.g., endoscopy) with microscopic analysis (e.g., histopathology) yields more robust predictions. However, fusing these modalities presents two major challenges:
1. **Spatial and Semantic Misalignment:** The modalities reside in vastly different feature spaces.
2. **Epistemic Uncertainty:** Standard softmax outputs are often overconfident, failing to capture the model's inherent lack of knowledge when encountering out-of-distribution or noisy data.

This project introduces **OT-Evidential Fusion**, an architecture that:
* Leverages **Optimal Transport (Sinkhorn-Knopp algorithm)** to compute a cross-modal transport plan, effectively aligning the latent representations of endoscopy and histopathology features.
* Projects these aligned features into an evidential framework based on **Subjective Logic**, transforming deterministic logits into Dirichlet belief masses.
* Utilizes **Dempster-Shafer Theory** to fuse the modalities dynamically based on their estimated epistemic uncertainty, yielding a final prediction accompanied by a quantifiable uncertainty mass ($u$).

## Repository Structure

multimodal-evidential-fusion/
├── requirements.txt           # Environment dependencies
├── utils/
│   ├── metrics.py             # Expected Calibration Error (ECE) and Brier Score calculations
│   └── plotting.py            # IEEE-style plotting (ROC, Confusion Matrices, Reliability Diagrams)
├── data/
│   └── dataset.py             # Multimodal data loaders (Kvasir v2 & LC25000)
├── models/
│   ├── backbones.py           # ResNet18 feature extractors
│   ├── baselines.py           # Comparative state-of-the-art fusion models (OT-CGSF, DDEF)
│   └── proposed.py            # OT-Evidential Fusion implementation
├── scripts/
│   └── train.py               # Standardized training and validation loop
├── main.py                    # Execution orchestrator
└── README.md
Baselines ImplementedTo rigorously validate the proposed architecture, the repository includes implementations of several state-of-the-art multimodal fusion techniques:OT-CGSF: Optimal Transport Cross-Guided Spatial Fusion.Resilient Sinkhorn: Logit-space fusion utilizing a resilient Sinkhorn algorithm.DDEF: Dual-Level Deep Evidential Fusion (pure subjective logic without optimal transport alignment).InstallationClone the repository:Bashgit clone [https://github.com/yourusername/multimodal-evidential-fusion.git](https://github.com/yourusername/multimodal-evidential-fusion.git)
cd multimodal-evidential-fusion
Install the required dependencies:Bashpip install -r requirements.txt
UsageEnsure your datasets (e.g., Kvasir v2 and LC25000) are located in the target directory (default is /kaggle/input). The data loader relies on identifying specific subdirectories (kvasir, colon_n, colon_aca) to establish binary classification targets (Benign vs. Malignant).To execute the entire pipeline (data loading, baseline and proposed model training, and analytical plotting), run the main orchestrator:Bashpython main.py
Outputs and VisualizationsDuring execution, the framework automatically generates and exports IEEE-formatted analytical figures to the ./paper_plots directory:Convergence Graphs: Epoch-wise training and validation loss curves.Confusion Matrices: Heatmaps detailing true positive and false positive rates.Receiver Operating Characteristic (ROC): Comparative Area Under the Curve (AUC) measurements.Reliability Diagrams: Model calibration curves plotted against the ideal $y=x$ line, bundled with Log-Scale Sharpness Histograms, Brier Scores, and Expected Calibration Error (ECE) metrics.
