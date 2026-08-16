# ✨ SPARK-UNet

**Sparse Prior-guided Attention with Region-aware Key-token sampling** for efficient 3D medical image segmentation.

Product implementation (variants **P / L / R**), aligned with manuscript `SPARK-UNet-final20260816.docx` (Figs 1–8, Tables 2–6).  
Figures synced from `D:\zhuomian\final-fig\` (**excluding** `three_datasets/` auxiliary plots).

[中文 README](README.md) · [Architecture flow](arch_flow.md) · [Asset sync](sync_paper_assets.py)

| Variant | Trainer (`-tr`) | Backbone | Legend |
|:-------:|:----------------|:---------|:-------|
| **P** | `nnUNetTrainerSPARKUNetP` | PlainConv | **Ours-P** (primary cross-dataset) |
| **L** | `nnUNetTrainerSPARKUNetL` | LightRes | **Ours-L** |
| **R** | `nnUNetTrainerSPARKUNetR` | ResEnc | **Ours-R** (ACDC emphasis) |

| Loss | Epochs |
|:-----|:-------|
| `L_seg + 0.05·L_prior + 0.02·PriorAux` | ACDC **200**; BraTS / Synapse **1000** (250 it/ep, SGD `1e-2`) |

**Paper ↔ code:** Reported checkpoints live under `nnUNetTrainerGLRPlainConv` / `GLRLightResConv` / `GLRResEncConv`. `nnUNetTrainerSPARKUNetP/L/R` is the clean product entry (`network.py`, `config.py`). Set **200** epochs for ACDC.

---

## 1. Architecture (Fig. 1)

Default **P5→S6**: **E0** Orient-only · **E1–E4** Read∥Win + BaOnBase · **E5** CNN + PriorHead (**no Bot**) · decoder **AG**.

<p align="center">
  <img src="assets/fig01_architecture.png" alt="Fig 1" width="920" />
</p>

<p align="center"><sub>
<strong>Fig. 1</strong> SPARK-UNet overview. (a) Full framework; (b) SPARKUnit; (c) AxialDW3D; (d) PriorHead; (e) TAN; (f) BA; (g) WinMHSA3D; (h) Decoder with AG.
</sub></p>

See [arch_flow.md](arch_flow.md) for stage-wise flow (paper Stage 1–6 ↔ code E0–E5).

---

## 2. Setup, training & inference

```bash
pip install -e .
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results

nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerSPARKUNetP
nnUNetv2_predict -i PATH/imagesTs -o PATH/pred -d DATASET_ID -c 3d_fullres -f all \
  -tr nnUNetTrainerSPARKUNetP -chk checkpoint_final.pth
```

---

## 3. Paper results (Figs 2–8 · Tables 2–6)

Held-out test: **ACDC** (40) · **BraTS2023-GLI** (251) · **Synapse** (12).  
CSV sources: [`table_summary.csv`](assets/table_summary.csv) · [`table6_efficiency.csv`](assets/table6_efficiency.csv) · [`complexity_paper_resource.csv`](assets/complexity_paper_resource.csv) · [`peak_mem_table.csv`](assets/peak_mem_table.csv)

### 3.0 Performance tables (overview)

#### Table A · Mean Dice (%↑) on three benchmarks

Matches the manuscript summary; `—` = not evaluated. **Ours-P/L/R** and **nnU-Net** in bold.

| Model | Params (M) | ACDC | BraTS | Synapse |
|:------|----------:|-----:|------:|--------:|
| **Ours-P** | 34.80 | 91.18 | **91.73** | **85.73** |
| **Ours-L** | 34.95 | 91.45 | 91.22 | 85.71 |
| **Ours-R** | 111.06 | **91.98** | 91.50 | 85.60 |
| **nnU-Net** | 31.20 | 91.17 | 91.36 | 84.43 |
| AttU-Net | 23.63 | 89.83 | 91.27 | 83.09 |
| CoTr | 41.93 | 88.66 | 90.97 | 85.35 |
| SegMamba | 66.90 | 89.37 | 90.99 | 81.68 |
| SegMamba-v2 | 138.78 | 90.76 | 91.21 | 82.49 |
| Swin-UNet | 30.64 | 85.18 | 89.54 | 73.54 |
| TransUNet | 113.26 | 89.59 | 91.15 | 82.36 |
| U-Mamba | 42.12 | 78.54 | 91.37 | 76.35 |
| UNETR | 130.79 | 82.22 | 89.94 | 73.28 |
| UNETR++ | 23.54 | 86.89 | 90.94 | 81.07 |
| LightUNETR-L | 3.93 | — | 90.36 | — |
| SlimUNETRV2 | 23.61 | — | 90.29 | — |

#### Table B · Efficiency & memory (Table 6 excerpt, per dataset)

RTX 4090. **FLOPs (G)** = plans-patch forward on each benchmark; **Latency (s)** = imagesTs sliding-window mean per case (fp32, no TTA); **Mem T / I (GB)** = train peak (AMP+backward, PyTorch allocated) / infer-patch peak (AMP forward, same patch). Matches Table 6 — **not** VRAM/SMI columns. Full 15-method Table 6: [`complexity_paper_resource.csv`](assets/complexity_paper_resource.csv) · [`peak_mem_table.csv`](assets/peak_mem_table.csv) (`train_peak_mem_GB` · `infer_patch_peak_mem_GB`).

| Model | Params (M) | ACDC FLOPs | ACDC Lat | ACDC Mem T/I | BraTS FLOPs | BraTS Lat | BraTS Mem T/I | Synapse FLOPs | Synapse Lat | Synapse Mem T/I |
|:------|----------:|-----------:|---------:|-------------:|------------:|----------:|--------------:|--------------:|------------:|----------------:|
| **Ours-P** | 34.80 | 226.7 | 0.062 | 12.4 / 0.8 | 575.1 | 0.377 | 9.4 / 1.7 | 664.7 | 13.143 | 13.0 / 2.6 |
| **Ours-L** | 34.95 | 243.1 | 0.120 | 13.7 / 0.9 | 634.1 | 0.418 | 10.8 / 2.0 | 732.5 | 10.260 | 12.3 / 2.0 |
| **Ours-R** | 111.06 | 440.4 | 0.071 | 15.0 / 1.2 | 925.5 | 0.487 | 11.5 / 2.1 | 1089.4 | 14.239 | 13.6 / 2.6 |
| **nnU-Net** | 31.20 | 203.4 | 0.111 | 6.2 / 0.6 | 538.1 | 0.340 | 6.2 / 1.7 | 619.6 | 5.883 | 7.3 / 2.5 |

Ours-P vs nnU-Net: **Params ~+11%**; FLOPs **+11.4% (ACDC) / +6.9% (BraTS) / +7.3% (Synapse)**; Latency varies by dataset and patch scale (ACDC **0.062 vs 0.111 s**; BraTS **0.377 vs 0.340 s**; Synapse **13.143 vs 5.883 s**).

#### Table C · Per-dataset highlights (Tables 2–4)

| Dataset | Table | Ours-P vs nnU-Net | Main takeaway |
|:--------|:------|:------------------|:--------------|
| BraTS | 2 | **91.73** vs 91.36 | TC **+0.77**, ET **+0.41**; WT ~flat |
| ACDC | 3 | **91.18** vs 91.17 | Ours-P ≈ nnU-Net; **Ours-R 91.98** |
| Synapse | 4 | **85.73** vs 84.43 | Gallbladder **72.47** vs 66.09 |

---

### 3.1 Fig. 2 · Dice–HD95 Pareto

<p align="center"><img src="assets/fig02_dice_hd95_pareto.png" alt="Fig 2" width="920" /></p>

<sub><strong>Fig. 2</strong> Mean DSC (↑) vs Mean HD95 (↓, mm) on three benchmarks; dashed Pareto fronts; near-front methods only. Left ACDC · middle BraTS · right Synapse.</sub>

### 3.2 Fig. 3 · Per-region Dice / HD95

<p align="center"><img src="assets/fig03_region_dice_hd95.png" alt="Fig 3" width="920" /></p>

<sub><strong>Fig. 3</strong> (a) region DSC; (b) region HD95. Labels vary by dataset (RV/MYO/LV · WT/TC/ET · 8 organs).</sub>

### 3.3 Fig. 4 · Qualitative segmentation

<p align="center"><img src="assets/fig04_qualitative.png" alt="Fig 4" width="920" /></p>

<sub><strong>Fig. 4</strong> Representative outputs: Ours-P on BraTS/Synapse, Ours-R on ACDC; GT vs prediction overlays.</sub>

### 3.4 Fig. 5 · Efficiency–accuracy (FLOPs–DSC)

<p align="center"><img src="assets/fig05_efficiency_flops_dsc.png" alt="Fig 5" width="920" /></p>

<p align="center"><sub><strong>Fig. 5</strong> FLOPs–DSC trade-off (★ Ours-P/L/R). Three panels left→right <strong>ACDC / BraTS / Synapse</strong>; x-axis = FLOPs (G, that dataset’s plans patch); y-axis = Mean DSC (%); labels show Params (M) and that dataset’s Latency (s). See <strong>Table B</strong>.</sub></p>

### 3.5 Fig. 6 · Case-level Mean Dice

<p align="center"><img src="assets/fig06_case_dice_distribution.png" alt="Fig 6" width="920" /></p>

<sub><strong>Fig. 6</strong> Per-case Mean Dice (strip + box). Methods sorted by test-set mean; box midline = median; <strong>top labels = same Mean Dice as Tables 2–4</strong>.</sub>

### 3.6 Fig. 7 · Prior / Top-K visualization (BraTS)

<p align="center">
  <img src="assets/fig07_prior_selection_viz.png" alt="Fig 7" width="920" />
</p>

<p align="center"><sub><strong>Fig. 7</strong> Single-case where-to-read on BraTS T1ce; Stages 2–5; TANOnly / Soft-Pr / Lock-Pr / PriorOnly / Random; axial / coronal / sagittal Top-K (red boxes).</sub></p>

### 3.7 Fig. 8 · Full held-out selection quality (BraTS)

<p align="center">
  <img src="assets/fig08_selection_quality.png" alt="Fig 8" width="920" />
</p>

<p align="center"><sub><strong>Fig. 8</strong> Selection quality on full held-out set (n=251; Stages 2–5): Precision@K, Recall@K, Enrichment, BoundaryHit@K by WT/TC/ET; mean ± 95% CI.</sub></p>

Ablation Tables 5a–5c are in the manuscript only.

---

## 4. Sync & verify assets

```bash
python nnunetv2/training/network/sparkunet/sync_paper_assets.py
python nnunetv2/training/network/sparkunet/verify_assets.py
```

Refreshes **8 manuscript figures + 4 CSV tables** from `D:\zhuomian\final-fig\`.

```text
sparkunet/
  README.md / README_EN.md / arch_flow.md
  sync_paper_assets.py / verify_assets.py
  assets/fig01…fig08 + table_*.csv
nnUNetTrainer/nnUNetTrainerSPARKUNet.py
```
