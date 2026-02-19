[README.md](https://github.com/user-attachments/files/25413500/README.md)
# 🎾 Tennis Stroke Classification via Skeleton-Based Deep Learning

## 📌 Project Overview
This project presents a Deep Learning system for classifying tennis strokes from video clips. Instead of applying computationally heavy Convolutional Neural Networks (CNNs) to raw image frames, this approach utilizes **Kinematic Skeleton Pose Estimation**. This methodology drastically reduces computational complexity and allows the model to focus purely on motion dynamics, completely ignoring background noise, lighting conditions, and player clothing.

The model was trained on the **THETIS dataset**, which includes thousands of video clips of amateur and professional players, classifying them into **12 distinct fine-grained classes** (e.g., Flat Service, Slice Service, Backhand Volley, etc.).

---

## 🎯 Key Challenges
1. **Fine-Grained Classification:** Distinguishing between highly similar movements (e.g., the subtle wrist difference between a Slice Serve and a Flat Serve).
2. **"Noisy" Amateur Videos:** Handling imperfect techniques, varying camera angles, and skeleton tracking noise.
3. **Variable Sequence Lengths:** Tennis strokes vary significantly in duration, requiring robust dynamic padding and masking mechanisms within the network.

---

## 🛠️ Data Pipeline & Engineering
Extensive data engineering was performed to ensure stable training and prevent biases:

* **MediaPipe Pose Extraction:** Extracting 33 joint landmarks per frame (X, Y, Z coordinates + Visibility score), resulting in 132 features per frame.
* **Sanity Checks & Smoothing:** * Tracking the hip center displacement to detect and filter out abnormal landmark jumps (Misdetections).
  * Applying a **Savitzky–Golay filter** to smooth jittery temporal movements while preserving natural human dynamics.
* **Normalization:** Normalizing spatial coordinates relative to the player's torso length, making the model invariant to player height or distance from the camera.
* **Data Augmentation (Mirroring):** Doubling the dataset size by horizontally flipping coordinates and swapping left/right joints (e.g., converting a Right-Handed Forehand to a Left-Handed Forehand).
* **Zero Data Leakage:** Implementing a strict Stratified Split (80/10/10) that guarantees an original video and its mirrored counterpart always reside in the same set (Train, Val, or Test).

---

## 🧠 Model Architectures: Baseline vs. Attention

Two deep recurrent architectures were evaluated:

### 1. Baseline Bi-GRU
A deep Bi-Directional GRU network (2 layers, Hidden Dimension: 128, Dropout: 0.3). 
This model leverages PyTorch's `pack_padded_sequence` to dynamically process variable-length sequences, ensuring the network only learns from actual data and explicitly ignores zero-padded frames. The final forward and backward hidden states are concatenated and passed through Linear and BatchNorm layers for classification.

### 2. Attention Bi-GRU (Additive / Bahdanau)
An Attention mechanism (Tanh-based) was integrated into the Bi-GRU to compute a Context Vector by assigning temporal weights to individual frames. 
* **Masking:** A strict boolean mask was applied, assigning extreme negative scores (`-1e9`) to padded frames before the Softmax operation, completely zeroing out their attention weights.

---

## 📊 Results & Research Findings

| Model | F1-Score (Validation) |
| :--- | :---: |
| **Baseline (Deep Bi-GRU)** | **81%** |
| Attention Bi-GRU | 76% |

The Baseline model achieved a remarkable **81% F1-score**, which is a highly robust result for fine-grained classification on noisy amateur video datasets. However, the most interesting discovery emerged from analyzing the Attention model's relative underperformance:

### 🔍 The Discovery: Kinematic Anticipation
An analysis of the confusion matrices revealed that the Attention model completely failed to classify the `slice_service`, frequently confusing it with the `flat_service`. 

Visualizing the live Attention weights showed that the network consistently peaked during the **Preparation phase (Backswing)** rather than the **Impact phase**. 
* **The Insight:** The model developed "Kinematic Anticipation." It attempted to classify the stroke based entirely on the player's initial stance and setup. Because the setup for a Slice Serve and a Flat Serve is nearly identical, the Attention mechanism made its decision "too early" and missed the critical wrist rotation at impact. The Baseline Bi-GRU, however, effectively utilized the entire temporal sequence memory to capture that final nuance.

---

## 💻 Technologies & Tools
* **Language:** Python 3.10
* **Deep Learning:** PyTorch (Dynamic Padding, Tensor Masking, `pack_padded_sequence`).
* **Computer Vision:** MediaPipe Pose, OpenCV.
* **Data Processing:** Pandas, NumPy.
* **Visualization:** Matplotlib, Seaborn.

---
