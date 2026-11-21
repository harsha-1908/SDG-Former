# SDG-Former: Scale–Depthwise–Gate Feed-Forward Networks in Hierarchical Vision Transformers

Official PyTorch implementation of **SDG-Former**, a hierarchical vision transformer that replaces uniform multilayer perceptron (MLP) feed-forward blocks with **Scale–Depthwise–Gate Feed-Forward Networks (SDG-FFN)** for robust natural and medical image classification.

This repository accompanies the paper:

**Scale–Depthwise–Gate Feed-Forward Networks in Hierarchical Vision Transformers for Robust Image Classification**

---

# ⚠️ Important Naming Clarification

This codebase retains its original directory structure, filenames, and internal class identifiers from earlier development (`DeKAHT` / `GN-KAN` / `swkat` / `GRKAN`). All CLI flags, filenames, and module imports remain unchanged so existing scripts run directly out of the box:

| Paper Terminology | Codebase Identifier |
| :--- | :--- |
| **SDG-Former** | `DeKAHT` (`dekaht.py`, model variants `swkat-tiny/small/base/large`) |
| **SDG-FFN** | `GN-KAN` / `GRKAN` (`--use-custom-kan`) |
| **SDG-Former-V1 (GELU)** | `gelu_variant/` |
| **SDG-Former-V2 (KANAct)** | `KAN_Act_variant/` |
| **Hybrid-V1 (W-MSA + SDG-FFN)** | `swmsa_mlp_wmsa_kan_hybrid_DEKAHT/` |
| **Hybrid-V2 (SW-MSA + SDG-FFN)** | `wmsa_mlp_swmsa_kan_hybrid_DEKAHT/` |

---

# 📂 Repository Structure

```text
.
├── KAN_Act_variant/
│   ├── dekaht.py                      # SDG-Former-V2 (KANAct activation)
│   └── train.py
│
├── gelu_variant/
│   ├── dekaht.py                      # SDG-Former-V1 (GELU activation)
│   └── train.py
│
├── hybrid_variants_for_ablation/
│   ├── swmsa_mlp_wmsa_kan_hybrid_DEKAHT/
│   │   ├── dekaht_hybrid.py           # Hybrid-V1 (W-MSA -> SDG-FFN, SW-MSA -> MLP)
│   │   └── train_dekaht_hybrid.py
│   │
│   └── wmsa_mlp_swmsa_kan_hybrid_DEKAHT/
│       ├── dekaht_hybrid_wmsa_kan.py  # Hybrid-V2 (W-MSA -> MLP, SW-MSA -> SDG-FFN)
│       └── train_dekaht_hybrid.py
│
├── ablation_results.md
├── example_run.sh
├── requirements.txt
└── README.md
