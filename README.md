# PE-AUAW-PINN-for-ICF-Implosion_github
# PE-AUAW-PINN: Physics-Informed Neural Networks for Extreme Multi-Scale ICF Implosion Modeling

Official PyTorch implementation of the paper: **"A Position-Encoded Physics-Informed Neural Network with Aleatoric Uncertainty-Based Adaptive Weighting for Extreme Multi-Scale ICF Implosion Modeling"** submitted to *Physics of Fluids*.

##  Overview

Inertial Confinement Fusion (ICF) implosion dynamics represent an extreme multi-scale problem, with physical quantities spanning up to 20 orders of magnitude. Standard Physics-Informed Neural Networks (PINNs) fail to converge due to severe spectral bias and gradient pathologies. 

This repository provides the **PE-AUAW-PINN** framework, which integrates:
1. **Positional Encoding (PE):** Mitigates spectral bias by mapping low-dimensional spatiotemporal coordinates into a higher-dimensional feature space, enabling the resolution of high-frequency shock discontinuities.
2. **Aleatoric Uncertainty-Based Adaptive Weighting (AUAW):** Dynamically balances the extremely disparate loss components (PDE residuals, initial/boundary conditions, and data fidelity) to stabilize multi-task training.

##  Requirements & Hardware

The code is tested on the following environment (as reported in the paper):
* **OS:** Linux / Windows 11
* **CPU:** AMD Ryzen 5 7500F (or equivalent)
* **GPU:** NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
* **Python:** 3.9+
* **PyTorch:** 2.0+ (CUDA enabled)
