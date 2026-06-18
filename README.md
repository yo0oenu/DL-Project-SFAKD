# Spatial-Frequency Attention for Knowledge Distillation

(2026-1 Computer Vision Term Project)

This project was conducted as part of a university Computer vision course.

---

## 1. Background

### 1.1 Knowledge Distillation

Knowledge Distillation (KD) is a model compression technique in which a small **student** network is trained to mimic the behavior of a larger, more powerful **teacher** network [1]. In the original formulation, the student is trained to match the teacher's softened output distribution (soft logits) via a temperature-scaled softmax, using KL divergence as the matching objective:

```
p_i = exp(z_i / T) / Σ_j exp(z_j / T)
```

While effective, relying solely on output-level (logit) supervision discards the rich structural information present in a teacher's intermediate representations.

### 1.2 Intermediate Feature Distillation

A complementary line of work distills knowledge from a teacher's **intermediate feature maps**, rather than (or in addition to) its final logits. Matching intermediate representations typically transfers more information than logit matching alone, since feature maps encode spatial and channel-wise structure learned at each stage of the network. Among feature-based methods, those that distill **attention** over the feature maps — i.e., emphasizing *where* and *what* the teacher focuses on — have shown particularly strong transfer performance, as the student is guided not just to replicate raw activations, but to replicate the teacher's salience.

### 1.3 Frequency Attention for KD (FAM-KD)

FAM-KD [2] argues that attention computed directly in the image (spatial) domain is inherently **local** — convolutional attention mechanisms aggregate information from limited receptive fields. To capture more **global** structure, FAM-KD computes attention in the **frequency domain** via the Fourier transform, where each frequency coefficient summarizes information across the entire spatial extent of the feature map.

Concretely, FAM-KD proposes a two-branch module:

- **Global branch:** FFT → Learnable Global Filter → High-Pass Filter (HPF) → IFFT, producing a frequency-attended feature map that captures global context.
- **Local branch:** a single 1×1 convolution, intended to preserve local detail.

The two branches are fused via a weighted sum (learnable scalars γ₁, γ₂) and the result is matched to the teacher's feature map (e.g., via MSE loss).

---

## 2. Motivation

While FAM-KD's global branch is well-motivated and effective at capturing long-range dependencies, its **local branch is underdeveloped**: a single 1×1 convolution has no spatial receptive field beyond a single pixel position (per channel), and therefore cannot model *spatial* relationships among neighboring regions. This leaves the "local information" half of the design under-powered relative to the sophistication of the frequency branch.

**Main idea of this project:** strengthen the local branch by replacing the 1×1 convolution with a module that explicitly computes **spatial attention** in the image domain, so that the local and global branches each meaningfully specialize in their respective domains before being fused.

---

## 3. Proposed Method

We retain FAM-KD's global (frequency) branch unchanged and redesign the local (spatial) branch, drawing on the spatial attention sub-module of **CBAM** (Convolutional Block Attention Module) [3].

### 3.1 Proposed Local Branch

Given an input feature map, the proposed local branch computes a spatial attention map as follows:

1. **Channel pooling:** apply both Global Average Pooling (GAP) and Global Max Pooling (GMP) along the channel dimension, producing two single-channel spatial maps.
2. **Concatenation:** concatenate the GAP and GMP maps along the channel axis.
3. **Convolution + Sigmoid:** pass the concatenated map through a convolutional layer followed by a sigmoid activation to obtain a spatial attention map: `Sigmoid(Conv(x))`.
4. This attention map is used to refine the (student) feature map by re-weighting spatially important regions.

This design gives the local branch an actual spatial receptive field and the ability to emphasize informative regions, which a bare 1×1 convolution cannot do.

### 3.2 Adaptive Fusion

As in FAM-KD, the outputs of the global (frequency) branch and the proposed local (spatial) branch are combined via a weighted sum using two **learnable scalars**:

```
output = γ1 · (global branch output) + γ2 · (proposed local branch output)
```

Allowing γ1 and γ2 to be learned (rather than fixed) lets the network adaptively balance the contribution of each domain per layer, which helps **prevent representational collapse** into a single dominant branch.

### 3.3 Training Objective

The fused student feature map is matched to the corresponding teacher feature map using an **MSE loss**, in addition to the standard classification loss against ground-truth labels (and optionally the original KD soft-logit loss).

---

## 4. Experiments

### 4.1 Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-100 |
| Teacher network | ResNet-56 |
| Student network | ResNet-20 |
| Task | Image classification |

All baselines and the proposed method were trained under identical settings (learning rate, number of epochs, optimizer, etc.) to ensure a fair comparison.

### 4.2 Results

| CIFAR-100 | ResNet-56 (Teacher) | ResNet-20 (Student, no KD) | KD (soft logit) [1] | FAM-KD [2] | Ours |
|---|---|---|---|---|---|
| Test Acc. | 73.23 | 69.37 | 70.66 | 71.39 | **71.82** |
| Best Acc. | 73.80 | 69.68 | 70.66 | 71.69 | **72.17** |

The proposed method outperforms vanilla logit-based KD and the original FAM-KD baseline, supporting the hypothesis that strengthening the local (spatial-attention) branch yields a more informative and better-balanced two-domain distillation signal.

---

## 5. Conclusion

- This project introduces a KD approach that combines **spatial attention** (image domain) with **frequency attention** (Fourier domain) for intermediate feature distillation.
- The two branches capture **complementary** information: local spatial cues from the image domain, and global structural cues from the frequency domain.
- **Learnable scalar fusion** (γ1, γ2) helps prevent the student representation from collapsing onto a single domain's signal.
- Overall, this work offers a perspective on how multi-domain attentive feature maps can be leveraged for more effective knowledge distillation.

---

## 6. References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531.
2. Pham, C., et al. (2024). *Frequency Attention for Knowledge Distillation.* Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV).
3. Woo, S., et al. (2018). *CBAM: Convolutional Block Attention Module.* Proceedings of the European Conference on Computer Vision (ECCV).
4. Ji, M., Heo, B., & Park, S. (2021). *Show, Attend and Distill: Knowledge Distillation via Attention-based Feature Matching.* Proceedings of the AAAI Conference on Artificial Intelligence.
5. Mansourian, A. M., Jalali, A., Ahmadi, R., & Kasaei, S. (2026). *Attention-Guided Feature Distillation for Semantic Segmentation.* Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) Workshops.
