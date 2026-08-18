# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — budgeted, anchor-selective multi-source contextual interaction for 3D medical image segmentation.

> 💡 **In one line:** prior picks **where to read**, Top-K caps **how many**, BA **writes back** cross-source context at anchors.

---

> ### 📌 Code & architecture availability
>
> **This page is a paper preview README.** The **implementation**, **training/inference configs**, and **full architecture documentation** matching the accepted manuscript will be **released in this repository** under `nnunetv2/training/network/sparkunet/` **upon formal paper acceptance**.
>

---

## Overview

SPARK-UNet injects long-range context on an nnU-Net-style **dense U-Net** trunk via **prior-guided where-to-read** and **budgeted anchor write-back (BA)**, instead of global self-attention over the full grid. Unselected voxels keep the CNN base, preserving dense outputs and sliding-window inference.

<p align="center">
  <img src="assets/fig01_architecture.png" alt="Fig 1" width="720" />
</p>

<p align="center"><sub><strong>Fig. 1</strong> SPARK-UNet architecture (preview). </sub></p>

---

## Citation

If this work is relevant to your research, please cite the paper (BibTeX / DOI will be added here upon acceptance).

---

## 📜 License

The released implementation is planned under the **[Apache License 2.0](LICENSE.txt)** after acceptance. Training/inference builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet); public benchmarks remain subject to their original terms.
