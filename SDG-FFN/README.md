# Scale–Depthwise–Gate Feed-Forward Networks (SDG-FFN) & SDG-Former

Official PyTorch implementation of **Scale–Depthwise–Gate Feed-Forward Networks in Hierarchical Vision Transformers for Robust Image Classification (SDG-FFN / SDG-Former)**.

> **Note on Repository Naming:** For backward compatibility with our earlier development release, the core operator module and repository files retain the internal identifier `GN-KAN` (`models/gnkan.py`, class `GNKAN`), which corresponds directly to the **SDG-FFN** architecture described in the paper.

SDG-FFN replaces uniform channel transformations in conventional Multilayer Perceptrons (MLPs) and Kolmogorov–Arnold Networks (KANs) with **disjoint group scaling ($\mathcal{S}$)**, **depthwise spatial refinement ($\mathcal{D}$)**, and **adaptive feature gating ($\mathcal{G}$)**.

---

## Architectural Overview

Given an input feature tensor $\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$, SDG-FFN executes the following sequential pipeline:

1. **Channel Expansion ($\mathbf{W}_e$)**
   - Projects input features to expanded channel capacity $D = rC$ ($r=4$ by default) via a $1\times 1$ pointwise convolution.
2. **Grouped Channel Scaling ($\mathcal{S}$)**
   - Partitions expanded channels into $N_g$ disjoint groups and modulates each subspace via independent learnable scaling coefficients $\boldsymbol{\alpha}^{(g)}$.
3. **Nonlinear Activation ($\phi$)**
   - Applies standard GELU (or learnable piecewise-linear `KANAct`) nonlinearity.
4. **Spatial Refinement ($\mathcal{D}$)**
   - Captures local spatial context via channel-independent $3\times 3$ depthwise convolution ($\text{DWConv}_{3\times 3}$).
5. **Adaptive Gating ($\mathcal{G}$)**
   - Filters background noise and highlights salient features via a $1\times 1$ sigmoid gating mask $\sigma(\mathbf{W}_g * \mathbf{X}_D) \odot \mathbf{X}_D$.
6. **Channel Projection ($\mathbf{W}_p$)**
   - Projects refined representations back to $C$ channels via a $1\times 1$ pointwise convolution for residual compatibility.

---

## Repository Structure

```text
GN-KAN/
│
├── experiments/
│   ├── exp01_mnist_ablation_100e.py
│   ├── exp02_fashionmnist_ablation_100e.py
│   ├── exp03_cifar10_ablation_100e.py
│   ├── exp04_braintumor_ablation_100e.py
│   ├── exp05_fashionmnist_ablation_100e.py
│   ├── exp06_cifar100_ablation_100e.py
│   ├── exp07_robustness_ablation.py
│   ├── exp12_data_efficiency_braintumor.py
│   ├── exp15_scaling_study_braintumor.py
│   └── exp16_efficiency_ablation.py
│
├── models/
│   ├── gnkan.py          # Core SDG-FFN operator implementation (class GNKAN)
│   ├── efficientkan.py   # Node-wise EfficientKAN baseline
│   ├── mlp.py            # Standard MLP baseline
│   └── sd.py             # Scale-Depthwise (SD) ablated variant
│
└── README.md
