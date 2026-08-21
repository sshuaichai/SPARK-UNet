"""
BraTS 测试集评估脚本（WT/TC/ET）：
计算每类 Dice、HD95 及最终平均值。

默认目录：
- root: D:\\zhuomian\\data\\nnUNet_raw\\Dataset1251_BraTS2023GLI
  - gt: <root>\\labelsTs
  - dataset.json: <root>\\dataset.json
  - pred: <root>\\labelsTs_\\*/{final_pre,best_pre}

加速:
  - HD95 使用 ROI 裁剪 + scipy EDT（替代 medpy 全卷计算）
  - 可选 --workers 多进程并行 case

输出：
- <pred_dir>\\metrics_BraTS_WT_TC_ET_testset.json
- <pred_dir>\\metrics_BraTS_WT_TC_ET_testset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

try:
    import SimpleITK as sitk
except ImportError as e:
    raise ImportError("需要 SimpleITK，请先安装: pip install SimpleITK") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


DEFAULT_ROOT = Path(r"D:\zhuomian\data\nnUNet_raw\Dataset1251_BraTS2023GLI")
DEFAULT_GT_DIR = DEFAULT_ROOT / "labelsTs"
DEFAULT_DATASET_JSON = DEFAULT_ROOT / "dataset.json"
METRIC_JSON = "metrics_BraTS_WT_TC_ET_testset.json"
METRIC_CSV = "metrics_BraTS_WT_TC_ET_testset.csv"
REGION_LABEL_IDS: Dict[str, Tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}


def _mask_in_labels(arr: np.ndarray, label_ids: Sequence[int]) -> np.ndarray:
    return np.isin(arr, np.asarray(label_ids, dtype=arr.dtype))


def _voxel_spacing_zyx(img: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()
    return float(sz), float(sy), float(sx)


def _to_np_zyx(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def _resample_label_to_reference(moving: sitk.Image, reference: sitk.Image) -> np.ndarray:
    if (
        moving.GetSize() == reference.GetSize()
        and moving.GetSpacing() == reference.GetSpacing()
        and moving.GetOrigin() == reference.GetOrigin()
        and moving.GetDirection() == reference.GetDirection()
    ):
        return _to_np_zyx(moving)

    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(reference)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    out = rs.Execute(moving)
    return _to_np_zyx(out)


def _dice_binary(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    s_p = int(p.sum())
    s_g = int(g.sum())
    if s_p == 0 and s_g == 0:
        return 1.0
    if s_p == 0 or s_g == 0:
        return 0.0
    inter = int(np.logical_and(p, g).sum(dtype=np.int64))
    return float(2.0 * inter / float(s_p + s_g))


def _binary_surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return mask
    eroded = binary_erosion(mask)
    if not np.any(eroded):
        return mask
    return mask & ~eroded


def _crop_union_bbox(
    *masks: np.ndarray,
    margin: int = 2,
) -> Tuple[np.ndarray, ...]:
    combined = masks[0].astype(bool, copy=False)
    for m in masks[1:]:
        combined = combined | m.astype(bool, copy=False)
    if not np.any(combined):
        return masks
    coords = np.nonzero(combined)
    slices = tuple(
        slice(max(0, int(c.min()) - margin), int(c.max()) + margin + 1)
        for c in coords
    )
    return tuple(m[slices] for m in masks)


def _hd95_binary(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing_zyx: Sequence[float]) -> Optional[float]:
    """ROI 裁剪 + 表面距离 EDT，比 medpy 全卷 HD95 快一个数量级以上。"""
    p = pred_bin.astype(bool, copy=False)
    g = gt_bin.astype(bool, copy=False)
    if not np.any(p) and not np.any(g):
        return 0.0
    if not np.any(p) or not np.any(g):
        return None

    p, g = _crop_union_bbox(p, g, margin=3)
    p_surf = _binary_surface(p)
    g_surf = _binary_surface(g)
    if not np.any(p_surf) or not np.any(g_surf):
        return None

    sampling = tuple(float(s) for s in spacing_zyx)
    dt_g = distance_transform_edt(~g, sampling=sampling)
    dt_p = distance_transform_edt(~p, sampling=sampling)
    dist_p_to_g = dt_g[p_surf]
    dist_g_to_p = dt_p[g_surf]
    if dist_p_to_g.size == 0 or dist_g_to_p.size == 0:
        return None
    return float(np.percentile(np.concatenate([dist_p_to_g, dist_g_to_p]), 95))


def _mean_skip_none(vals: List[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in vals if v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if not xs:
        return None
    return float(np.mean(xs))


def _round2(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return round(float(v), 2)


def _to_percent(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    fv = float(v)
    if 0.0 <= fv <= 1.0:
        return fv * 100.0
    return fv


def evaluate_one_case(
    gt_path: Path,
    pred_path: Path,
    compute_hd95: bool = True,
) -> Dict[str, Any]:
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))
    gt_arr = _to_np_zyx(gt_img)
    pred_arr = _resample_label_to_reference(pred_img, gt_img)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"shape不一致: {gt_path.name}, gt={gt_arr.shape}, pred={pred_arr.shape}")

    spacing = _voxel_spacing_zyx(gt_img)
    case_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    case_dice_vals: List[Optional[float]] = []
    case_hd_vals: List[Optional[float]] = []

    for rname, lids in REGION_LABEL_IDS.items():
        pb = _mask_in_labels(pred_arr, lids)
        gb = _mask_in_labels(gt_arr, lids)
        d = _to_percent(_dice_binary(pb, gb))
        h = _hd95_binary(pb, gb, spacing) if compute_hd95 else None
        d = _round2(d)
        h = _round2(h)
        case_metrics[rname] = {"Dice": d, "HD95": h}
        case_dice_vals.append(d)
        case_hd_vals.append(h)

    return {
        "case": gt_path.name,
        "metrics": case_metrics,
        "Mean_Dice_WT_TC_ET": _round2(_mean_skip_none(case_dice_vals)),
        "Mean_HD95_WT_TC_ET": _round2(_mean_skip_none(case_hd_vals)) if compute_hd95 else None,
    }


def evaluate(
    gt_dir: Path,
    pred_dir: Path,
    workers: int = 1,
    compute_hd95: bool = True,
) -> Dict[str, Any]:
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"GT目录无 nii.gz: {gt_dir}")

    per_case: List[Dict[str, Any]] = []

    if workers <= 1:
        it = gt_files
        if tqdm is not None:
            it = tqdm(gt_files, desc=pred_dir.name, unit="case")
        for gt_path in it:
            pred_path = pred_dir / gt_path.name
            if not pred_path.is_file():
                raise FileNotFoundError(f"缺少预测文件: {pred_path}")
            per_case.append(evaluate_one_case(gt_path, pred_path, compute_hd95=compute_hd95))
    else:
        tasks = []
        for gt_path in gt_files:
            pred_path = pred_dir / gt_path.name
            if not pred_path.is_file():
                raise FileNotFoundError(f"缺少预测文件: {pred_path}")
            tasks.append((gt_path, pred_path))

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(evaluate_one_case, gt_path, pred_path, compute_hd95): gt_path.name
                for gt_path, pred_path in tasks
            }
            done_map: Dict[str, Dict[str, Any]] = {}
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc=pred_dir.name, unit="case")
            for fut in iterator:
                result = fut.result()
                done_map[result["case"]] = result
        per_case = [done_map[gt_path.name] for gt_path in gt_files]

    per_region_dice: Dict[str, List[Optional[float]]] = {k: [] for k in REGION_LABEL_IDS}
    per_region_hd95: Dict[str, List[Optional[float]]] = {k: [] for k in REGION_LABEL_IDS}
    for case_result in per_case:
        for rname in REGION_LABEL_IDS:
            m = case_result["metrics"][rname]
            per_region_dice[rname].append(m["Dice"])
            per_region_hd95[rname].append(m["HD95"])

    mean_block: Dict[str, Dict[str, Optional[float]]] = {}
    wt_tc_et_dice: List[Optional[float]] = []
    wt_tc_et_hd: List[Optional[float]] = []
    for rname in ("WT", "TC", "ET"):
        dm = _round2(_mean_skip_none(per_region_dice[rname]))
        hm = _round2(_mean_skip_none(per_region_hd95[rname])) if compute_hd95 else None
        mean_block[rname] = {"Dice": dm, "HD95": hm}
        wt_tc_et_dice.append(dm)
        wt_tc_et_hd.append(hm)

    return {
        "meta": {
            "schema": "BraTS_WT_TC_ET_testset",
            "regions": {k: list(v) for k, v in REGION_LABEL_IDS.items()},
            "gt_dir": str(gt_dir),
            "pred_dir": str(pred_dir),
            "n_cases": len(per_case),
        },
        "mean": mean_block,
        "foreground_mean": {
            "Dice": _round2(_mean_skip_none(wt_tc_et_dice)),
            "HD95": _round2(_mean_skip_none(wt_tc_et_hd)) if compute_hd95 else None,
        },
        "metric_per_case": per_case,
    }


def write_outputs(summary: Dict[str, Any], pred_dir: Path) -> None:
    out_json = pred_dir / METRIC_JSON
    out_csv = pred_dir / METRIC_CSV

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows = []
    for rname in ("WT", "TC", "ET"):
        m = summary["mean"].get(rname, {})
        rows.append(
            {
                "Region": rname,
                "Dice_pct": m.get("Dice", ""),
                "HD95_mm": m.get("HD95", ""),
            }
        )
    rows.append(
        {
            "Region": "Mean_WT_TC_ET",
            "Dice_pct": summary["foreground_mean"].get("Dice", ""),
            "HD95_mm": summary["foreground_mean"].get("HD95", ""),
        }
    )

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["Region", "Dice_pct", "HD95_mm"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"已写入: {out_json}")
    print(f"已写入: {out_csv}")


def outputs_exist(pred_dir: Path) -> bool:
    return (pred_dir / METRIC_JSON).exists() and (pred_dir / METRIC_CSV).exists()


def discover_pred_dirs(root: Path) -> List[Path]:
    pred_dirs: List[Path] = []
    for model_dir in sorted(root.glob("labelsTs_*")):
        if not model_dir.is_dir():
            continue
        for sub in ("final_pre", "best_pre"):
            p = model_dir / sub
            if p.is_dir():
                pred_dirs.append(p)
    return pred_dirs


def main() -> None:
    ap = argparse.ArgumentParser(description="评估 BraTS 测试集 WT/TC/ET 的 Dice 与 HD95")
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="数据集根目录")
    ap.add_argument("--gt", type=str, default=str(DEFAULT_GT_DIR), help="GT目录，默认 <root>/labelsTs")
    ap.add_argument("--pred-dirs", type=str, nargs="+", default=None, help="一个或多个预测目录（提供后将覆盖自动发现）")
    ap.add_argument("--auto-discover", action="store_true", help="自动扫描 <root>/labelsTs_*/{final_pre,best_pre}")
    ap.add_argument(
        "--dataset-json",
        type=str,
        default=str(DEFAULT_DATASET_JSON),
        help="dataset.json 路径（当前脚本不依赖该文件参与计算，仅用于路径一致性检查）",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="即使评估结果已存在也强制重算并覆盖",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="并行评估 case 的进程数（默认 min(8, CPU核数)）",
    )
    ap.add_argument(
        "--dice-only",
        action="store_true",
        help="仅计算 Dice，跳过 HD95（最快）",
    )
    args = ap.parse_args()

    root = Path(args.root)
    gt_dir = Path(args.gt)
    dataset_json = Path(args.dataset_json)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT目录不存在: {gt_dir}")
    if not dataset_json.is_file():
        raise FileNotFoundError(f"dataset.json 不存在: {dataset_json}")

    if args.pred_dirs:
        pred_dirs = [Path(p) for p in args.pred_dirs]
    else:
        pred_dirs = discover_pred_dirs(root)
    if not pred_dirs:
        raise RuntimeError("未发现可评估的预测目录。可用 --pred-dirs 显式传入。")

    print(f"root: {root}")
    print(f"gt: {gt_dir}")
    print(f"dataset.json: {dataset_json}")
    print(f"并行 workers={args.workers}, HD95={'关闭' if args.dice_only else '开启(EDT+ROI)'}")

    pred_iter = pred_dirs
    if tqdm is not None:
        pred_iter = tqdm(pred_dirs, desc="pred_dirs", unit="dir")

    for pred_dir in pred_iter:
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"预测目录不存在: {pred_dir}")
        if outputs_exist(pred_dir) and not args.force:
            print(f"\n跳过已评估目录: {pred_dir}（已存在 {METRIC_JSON}/{METRIC_CSV}）")
            continue

        print(f"\n开始评估: {pred_dir}")
        summary = evaluate(
            gt_dir,
            pred_dir,
            workers=args.workers,
            compute_hd95=not args.dice_only,
        )
        write_outputs(summary, pred_dir)

        fg = summary["foreground_mean"]
        print(f"[{pred_dir.parent.name}/{pred_dir.name}] Mean_WT_TC_ET: Dice={fg.get('Dice')}%, HD95={fg.get('HD95')} mm")


if __name__ == "__main__":
    main()
