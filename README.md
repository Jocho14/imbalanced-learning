# CIFAR-10 Imbalanced Learning: Advanced Strategies

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

This repository implements state-of-the-art techniques for training Deep Neural Networks on highly imbalanced datasets. It uses a custom implementation of **ResNet-18** on a synthetically imbalanced **CIFAR-10** dataset (Step Imbalance) to simulate real-world long-tailed recognition scenarios.

## 📉 The Challenge: Step Imbalance
The project simulates an extreme imbalance scenario where one "normal" class dominates, and all other classes are scarce.
* **Majority Class:** 5000 samples (Class Index 3)
* **Minority Classes:** 150 samples each (Ratio 0.03)
* **Imbalance Ratio:** ~33:1

## 🧠 Implemented Strategies

The repository includes modular trainers implementing various papers and techniques:

### 1. **Expert Mode (LDAM-DRW)**
*Based on "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)*
* **Loss:** **LDAM Loss** with a strong margin (`s=30`) pushes decision boundaries away from minority classes.
* **Sampling:** Weighted Random Sampling to balance batch statistics.
* **Scheduling:** **Deferred Re-Weighting (DRW)**. The model trains without loss-reweighting initially to learn features, then switches to re-weighted loss (or sampler) in later epochs (e.g., epoch 160).

### 2. **RotNet (Self-Supervised Pre-training)**
*Based on "Self-Supervised Learning for Imbalanced Data"*
* **Phase 1 (SSL):** The model is trained to predict the rotation of an image (0°, 90°, 180°, 270°) rather than its class. This forces the model to learn semantic features without being biased by class frequency.
* **Phase 2 (Fine-Tuning):** The linear head is reset, and the model is fine-tuned on the classification task using a balanced sampler.

### 3. **Hybrid Mode**
Combines multiple modern tricks for maximum performance:
* **NormedLinear:** Normalizes weights in the final layer to prevent the magnitude of majority classes from dominating.
* **MixUp:** Convex combination of input pairs and labels (Always ON) to regularize the manifold.
* **LDAM Loss:** With `s=10` and `max_m=0.5`.

### 4. **Pro Mode**
A robust baseline using standard industry tricks:
* **Weighted Random Sampler:** Ensures each batch has roughly equal class representation.
* **MixUp Augmentation:** Heavily regularizes the network.
* **Label Smoothing:** Prevents overconfidence on majority classes.
