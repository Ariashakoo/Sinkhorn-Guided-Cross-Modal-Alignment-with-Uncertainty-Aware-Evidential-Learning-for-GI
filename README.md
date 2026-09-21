# Sinkhorn-Guided Cross-Modal Alignment with Uncertainty-Aware Evidential Learning for GI

Implementation of the paper:

**"Sinkhorn Guided Cross Modal Alignment with Uncertainty Aware Evidential Learning for Gastrointestinal Cancer Detection"**

This repository contains the PyTorch implementation of a novel multimodal fusion architecture designed for gastrointestinal medical image classification. The framework combines the spatial alignment capabilities of Sinkhorn Optimal Transport with the uncertainty-awareness of Subjective Logic and Evidential Deep Learning to fuse Endoscopy and Histopathology imagery.

---

## Project Overview


In medical diagnostics, combining macroscopic findings (e.g., endoscopy) with microscopic analysis (e.g., histopathology) yields more robust predictions. However, fusing these modalities presents two major challenges:
1. **Spatial and Semantic Misalignment:** The modalities reside in vastly different feature spaces.
2. **Epistemic Uncertainty:** Standard softmax outputs are often overconfident, failing to capture the model's inherent lack of knowledge when encountering out-of-distribution or noisy data.

This project introduces **OT-Evidential Fusion**, an architecture that:
* Leverages **Optimal Transport (Sinkhorn-Knopp algorithm)** to compute a cross-modal transport plan, effectively aligning the latent representations of endoscopy and histopathology features.
* Projects these aligned features into an evidential framework based on **Subjective Logic**, transforming deterministic logits into Dirichlet belief masses.
* Utilizes **Dempster-Shafer Theory** to fuse the modalities dynamically based on their estimated epistemic uncertainty, yielding a final prediction accompanied by a quantifiable uncertainty mass ($u$).

## Repository Structure
```
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
```
Baselines ImplementedTo rigorously validate the proposed architecture, the repository includes implementations of several state-of-the-art multimodal fusion techniques:OT-CGSF: Optimal Transport Cross-Guided Spatial Fusion.Resilient Sinkhorn: Logit-space fusion utilizing a resilient Sinkhorn algorithm.DDEF: Dual-Level Deep Evidential Fusion (pure subjective logic without optimal transport alignment).InstallationClone the repository:Bashgit clone [https://github.com/Ariashakoo/Sinkhorn-Guided-Cross-Modal-Alignment-with-Uncertainty-Aware-Evidential-Learning-for-GI.git](https://github.com/Ariashakoo/Sinkhorn-Guided-Cross-Modal-Alignment-with-Uncertainty-Aware-Evidential-Learning-for-GI.git)
cd multimodal-evidential-fusion
Install the required dependencies:Bashpip install -r requirements.txt
UsageEnsure your datasets (e.g., Kvasir v2 and LC25000) are located in the target directory (default is /kaggle/input). The data loader relies on identifying specific subdirectories (kvasir, colon_n, colon_aca) to establish binary classification targets (Benign vs. Malignant).To execute the entire pipeline (data loading, baseline and proposed model training, and analytical plotting), run the main orchestrator:Bashpython main.py
Outputs and VisualizationsDuring execution, the framework automatically generates and exports IEEE-formatted analytical figures to the ./paper_plots directory:Convergence Graphs: Epoch-wise training and validation loss curves.Confusion Matrices: Heatmaps detailing true positive and false positive rates.Receiver Operating Characteristic (ROC): Comparative Area Under the Curve (AUC) measurements.Reliability Diagrams: Model calibration curves plotted against the ideal $y=x$ line, bundled with Log-Scale Sharpness Histograms, Brier Scores, and Expected Calibration Error (ECE) metrics.

In medical diagnostics, combining macroscopic findings, such as endoscopy, with microscopic analysis, such as histopathology, can provide complementary information for gastrointestinal cancer detection.

However, fusing these modalities presents two major challenges:

1. **Spatial and Semantic Misalignment:** Endoscopy and histopathology modalities reside in substantially different feature spaces, making direct feature fusion challenging.

2. **Epistemic Uncertainty:** Standard softmax-based neural networks can produce highly confident predictions even when the input is noisy, ambiguous, or outside the model's learned distribution.

This project introduces **OT-Evidential Fusion**, an architecture designed to address both challenges.

The proposed framework:

- Leverages **Optimal Transport with the Sinkhorn-Knopp algorithm** to compute a cross-modal transport plan and align latent representations between endoscopy and histopathology features.
- Projects the aligned representations into an **Evidential Deep Learning** framework based on **Subjective Logic**, transforming deterministic predictions into evidence-based representations.
- Utilizes **Dempster-Shafer Theory** to dynamically fuse evidence from the different modalities while accounting for their estimated epistemic uncertainty.
- Produces a final classification prediction together with a quantifiable uncertainty mass.

The overall goal is to develop a multimodal medical image classification framework that considers not only the predicted class but also the reliability and uncertainty associated with the prediction.

---

## Motivation

Endoscopy and histopathology provide complementary views of gastrointestinal disease.

Endoscopic images capture the macroscopic appearance of gastrointestinal lesions, while histopathology images provide microscopic information about tissue and cellular structures.

A conventional single-modality classifier can be represented as:

```text
Medical Image
      |
      v
Feature Extractor
      |
      v
Classifier
      |
      v
Prediction
