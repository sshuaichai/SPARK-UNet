# ✨Spark-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — budgeted, anchor-selective multi-source contextual interaction for 3D medical image segmentation.

[中文 README](README.md)

---

> ### 📌 Code & architecture availability
>
> The **implementation**, **training/inference configs**, and **full architecture documentation** matching the accepted manuscript will be **released in this repository upon formal paper acceptance**.

---

## Overview

SPARK-UNet injects long-range context on an nnU-Net-style **dense U-Net** trunk via **prior-guided where-to-read** and **budgeted anchor write-back (BA)**, instead of global self-attention over the full grid. Unselected voxels keep the CNN base, preserving dense outputs and sliding-window inference.


---

## Citation

If this work is relevant to your research, please cite the paper (BibTeX / DOI will be added here upon acceptance).

---

## 📜 License

The implementation will be released after acceptance, planned as **[Apache License 2.0](LICENSE.txt)**. Training/inference builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet); public benchmarks remain subject to their original terms.

---

## Data citation & downloads (nnU-Net format)

Cloud links below provide **nnU-Net–format** dataset packs. Official portals are for citation and licensing. Follow each dataset’s original terms of use.

### ACDC

| Type | Link |
|:-----|:-----|
| nnU-Net pack (Baidu Netdisk) | [ACDC](https://pan.baidu.com/s/1UpbyOIFCrYgThEsCaDyAWg?pwd=fr7t) · code `fr7t` |
| nnU-Net pack (Aliyun Drive) | [ACDC](https://www.alipan.com/s/EJPiXceGWZV) |
| Official source | [Human Heart Project / ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| TransUNet-split preprocessed reference | [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4) |

### Synapse / BTCV

| Type | Link |
|:-----|:-----|
| nnU-Net pack (Baidu Netdisk) | [Synapse](https://pan.baidu.com/s/1IvX_5Q1h6QeSDa__gjEX_A?pwd=drsm) · code `drsm` |
| Official source (BTCV / Synapse) | [Synapse: syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| TransUNet-split preprocessed reference | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

### BraTS 2021 Adult Glioma

Paper and experiments use the **BraTS 2021** publicly labeled training cohort (1251 cases). BraTS 2022/2023 Adult Glioma redistributed the same cohort and are **not** BraTS 2025 Lighthouse.

| Type | Link |
|:-----|:-----|
| nnU-Net pack (Aliyun Drive) | [Dataset1251_BraTS2021GLI](https://www.alipan.com/s/M7cS2KvaAuK) |
| Official source (BraTS 2021) | [Synapse: syn25829067](https://www.synapse.org/Synapse:syn25829067) |
| Same-cohort redistribute page (2023 challenge) | [Synapse: syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351) |
| Kaggle mirror (same cohort) | [part-1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) · [part-2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip) |
