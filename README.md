# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — budgeted, anchor-selective multi-source contextual interaction for 3D medical image segmentation.

> 💡 **In one line:** prior picks **where to read**, Top-K caps **how many**, BA **writes back** cross-source context at anchors.

[中文 README](README.md) · [Full README (figures & tables)](README_EN.full.md)

---

> ### 📌 Code & architecture availability
>
> **This page is a paper preview README.** The **implementation**, **training/inference configs**, and **full architecture documentation** matching the accepted manuscript will be **released in this repository** under `nnunetv2/training/network/sparkunet/` **upon formal paper acceptance**.
>
> Before acceptance, this directory provides a **method overview** only. Benchmark metrics, full figures, and reproduction details are in [README_EN.full.md](README_EN.full.md) (internal sync with the manuscript).

---

## Overview

SPARK-UNet injects long-range context on an nnU-Net-style **dense U-Net** trunk via **prior-guided where-to-read** and **budgeted anchor write-back (BA)**, instead of global self-attention over the full grid. Unselected voxels keep the CNN base, preserving dense outputs and sliding-window inference.

**SPARKUnit pipeline (E1–E4):**

| Step | Module | Role |
|:----:|:-------|:-----|
| 1 | **Orient (E0)** | AxialDW3D local orientation |
| 2 | **Prior + TAN** | PriorHead → P_eff; Top-K on **h** → **base** |
| 3 | **BA ∥ Win** | BA writes to **base** at anchors (history / global / partial; softmax over source dim \(S\)); WinMHSA3D on **h** adds in parallel via a gate |

Default **pool=P5 → stages=S6** (E0–E5); decoder with **AG** attention gates. Variants: **Ours-P** (PlainConv, primary) · **Ours-L** (LightRes) · **Ours-R** (ResEnc-L).

<p align="center">
  <img src="assets/fig01_architecture.png" alt="Fig 1" width="720" />
</p>

<p align="center"><sub><strong>Fig. 1</strong> SPARK-UNet architecture (preview). Full module labels ship with the released code and [README_EN.full.md](README_EN.full.md).</sub></p>

---

## Citation

If this work is relevant to your research, please cite the paper (BibTeX / DOI will be added here upon acceptance).

---

## 📜 License

The released implementation is planned under the **[Apache License 2.0](LICENSE.txt)** after acceptance. Training/inference builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet); public benchmarks remain subject to their original terms.
