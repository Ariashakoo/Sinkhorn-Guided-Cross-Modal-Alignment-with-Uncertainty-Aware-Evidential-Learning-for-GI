# Sinkhorn-Guided Cross-Modal Alignment with Uncertainty-Aware Evidential Learning for GI

Implementation of the paper:

**"Sinkhorn Guided Cross Modal Alignment with Uncertainty Aware Evidential Learning for Gastrointestinal Cancer Detection"**

This repository contains the PyTorch implementation of a novel multimodal fusion architecture designed for gastrointestinal medical image classification. The framework combines the spatial alignment capabilities of Sinkhorn Optimal Transport with the uncertainty-awareness of Subjective Logic and Evidential Deep Learning to fuse Endoscopy and Histopathology imagery.

---

## Project Overview

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
