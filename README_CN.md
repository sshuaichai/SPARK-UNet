# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — 预算约束的锚点选择性多源上下文交互，用于三维医学图像分割。

[English README](README_EN.md)

---

> ### 📌 代码与架构公开说明
>
> 与正式稿件一致的可复现 **实现代码**、**训练/推理配置** 及 **完整架构说明**，将在论文 **被期刊/会议正式接受（acceptance）后** 于本仓库公开。

---

## 方法概览

SPARK-UNet 在 nnU-Net 式 **稠密 U-Net** 主干上，用 **先验引导的 where-to-read** 与 **预算约束的 anchor write-back（BA）** 注入长程上下文，避免对全网格做全局自注意力；未选中位置保持 CNN 基底，仍支持稠密输出与滑窗推理。

<p align="center">
  <img src="assets/fig01_architecture.png" alt="Fig 1" width="720" />
</p>

<p align="center"><sub><strong>图 1</strong> SPARK-UNet 架构示意。</sub></p>

---

## 引用

若本工作与您的研究相关，请引用正式论文（接受后将在此补充 BibTeX / DOI）。

---

## 📜 License

方法实现将于接受后发布，计划以 **[Apache License 2.0](LICENSE.txt)** 授权。训练/推理基于 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)；公开数据集须遵守各自使用条款。

---

## 数据引用与下载（nnU-Net 格式）

下列网盘提供 **已转换为 nnU-Net 格式** 的数据包；官方源站用于引用与授权。使用前请遵守各数据集原始条款。

### ACDC

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [ACDC](https://pan.baidu.com/s/1UpbyOIFCrYgThEsCaDyAWg?pwd=fr7t) · 提取码 `fr7t` |
| nnU-Net 格式（阿里云盘） | [ACDC](https://www.alipan.com/s/EJPiXceGWZV) |
| 官方源 | [Human Heart Project / ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| TransUNet 划分预处理参考 | [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4) |

### Synapse / BTCV

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [Synapse](https://pan.baidu.com/s/1IvX_5Q1h6QeSDa__gjEX_A?pwd=drsm) · 提取码 `drsm` |
| 官方源（BTCV / Synapse） | [Synapse: syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| TransUNet 划分预处理参考 | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

### BraTS 2021 Adult Glioma

论文与实验口径为 **BraTS 2021** 公开带标注训练队列（1251 例）。BraTS 2022/2023 Adult Glioma 为同一队列的再分发，**不等于** BraTS 2025 Lighthouse。

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（阿里云盘） | [Dataset1251_BraTS2021GLI](https://www.alipan.com/s/M7cS2KvaAuK) |
| 官方源（BraTS 2021） | [Synapse: syn25829067](https://www.synapse.org/Synapse:syn25829067) |
| 同队列再分发参考（2023 challenge 页） | [Synapse: syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351) |
| Kaggle 镜像（同队列） | [part-1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) · [part-2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip) |
