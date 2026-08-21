# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** — 预算约束的锚点选择性多源上下文交互，用于三维医学图像分割。

> 💡 **一句话：** 先验定 **读哪里**，Top-K 限 **读多少**，BA **写回** 跨源上下文 —— 未选中体素保持恒等。

[English README](README_EN.md)

---

> ### 📌 代码与架构公开说明
>
> 与正式稿件一致的可复现 **实现代码**、**训练/推理配置** 及 **完整架构说明**，将在论文 **被期刊/会议正式接受（acceptance）后**，于 **本代码库** 的 `nnunetv2/training/network/sparkunet/` 路径公开。

---

## 方法概览

SPARK-UNet 在 nnU-Net 式 **稠密 U-Net** 主干上，用 **先验引导的 where-to-read** 与 **预算约束的 anchor write-back（BA）** 注入长程上下文，避免对全网格做全局自注意力；未选中位置保持 CNN 基底，仍支持稠密输出与滑窗推理。

**SPARKUnit 流水线（E1–E4）：**

| 步骤 | 模块 | 作用 |
|:----:|:-----|:-----|
| 1 | **Orient（E0）** | AxialDW3D 轴对齐局部编码 |
| 2 | **Prior + TAN** | PriorHead → P_eff；在 **h** 上 Top-K → **base** |
| 3 | **BA ∥ Win** | BA 在锚点写回 **base**（history / 全局 / partial，源维 \(S\) softmax）；WinMHSA3D 在 **h** 上门控并联相加 |

默认 **pool=P5 → stages=S6**（E0–E5）；解码器含 **AG** 注意力门。变体：**Ours-P**（PlainConv，主报告）· **Ours-L**（LightRes）· **Ours-R**（ResEnc-L）。

<p align="center">
  <img src="assets/fig01_architecture.png" alt="Fig 1" width="720" />
</p>

<p align="center"><sub><strong>图 1</strong> SPARK-UNet 架构示意。</sub></p>

---

## 引用

若本工作与您的研究相关，请引用正式论文（接受后将在此补充 BibTeX / DOI）。

---

## 📜 License

方法实现将于接受后在本路径发布，计划以 **[Apache License 2.0](LICENSE.txt)** 发布。训练/推理基于 [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)；公开数据集须遵守各自使用条款。

---

## 数据引用与下载（nnU-Net 格式）

下列网盘包为 **已转换的 nnU-Net raw 目录**（`Dataset*`）；官方源站用于引用与授权。使用前请遵守各数据集原始条款。

### ACDC · `Dataset100_ACDC`

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [Dataset100_ACDC](https://pan.baidu.com/s/1UpbyOIFCrYgThEsCaDyAWg?pwd=fr7t) · 提取码 `fr7t` |
| nnU-Net 格式（阿里云盘） | [Dataset100_ACDC](https://www.alipan.com/s/EJPiXceGWZV) |
| 官方源 | [Human Heart Project / ACDC](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) |
| TransUNet 划分预处理参考 | [Google Drive](https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4) |

### Synapse / BTCV · `Dataset180_Synapse`

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（百度网盘） | [Dataset180_Synapse](https://pan.baidu.com/s/1IvX_5Q1h6QeSDa__gjEX_A?pwd=drsm) · 提取码 `drsm` |
| 官方源（BTCV / Synapse） | [Synapse: syn3193805](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |
| TransUNet 划分预处理参考 | [Google Drive](https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd) |

### BraTS2023-GLI · `Dataset1251_BraTS2023GLI`

| 类型 | 链接 |
|:-----|:-----|
| nnU-Net 格式（阿里云盘） | [Dataset1251_BraTS2023GLI](https://www.alipan.com/s/XsuCtxNFDaz) |
| 官方源 | [Synapse: syn51156910](https://www.synapse.org/Synapse:syn51156910/wiki/622351) |
| Kaggle 镜像 | [part-1](https://www.kaggle.com/datasets/aiocta/brats2023-part-1) · [part-2](https://www.kaggle.com/datasets/aiocta/brats2023-part-2zip) |
