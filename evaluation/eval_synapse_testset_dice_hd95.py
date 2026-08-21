"""
评估 Synapse 测试集：Dice + HD95（每类 + 平均）

默认评估目录：
- GT:    D:\\zhuomian\\data\\nnUNet_raw\\Dataset180_Synapse\\labelsTs
- Pred:  <root>\\labelsTs_\\*/{final_pre,best_pre}

加速:
  - HD95 使用 ROI 裁剪 + scipy EDT（替代 medpy 全卷计算）
  - 可选 --workers 多进程并行 case

输出：
- <pred_dir>\\metrics_synapse_testset.json
- <pred_dir>\\metrics_synapse_testset.csv
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


DEFAULT_ROOT = Path(r"D:\zhuomian\data\nnUNet_raw\Dataset180_Synapse")
DEFAULT_GT_DIR = DEFAULT_ROOT / "labelsTs"
DEFAULT_DATASET_JSON = DEFAULT_ROOT / "dataset.json"
METRIC_JSON = "metrics_synapse_testset.json"
METRIC_CSV = "metrics_synapse_testset.csv"


def load_label_map(dataset_json: Path) -> List[Tuple[int, str]]:
    with open(dataset_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = data.get("labels", {})
    items = []
    for name, lid in labels.items():
        lid_i = int(lid)
        if lid_i == 0:
            continue
        items.append((lid_i, str(name)))
    items.sort(key=lambda x: x[0])
    return items


def voxel_spacing_zyx(img: sitk.Image) -> Tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()
    return float(sz), float(sy), float(sx)


def to_np_zyx(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def resample_pred_to_gt(pred: sitk.Image, gt: sitk.Image) -> np.ndarray:
    if (
        pred.GetSize() == gt.GetSize()
        and pred.GetSpacing() == gt.GetSpacing()
        and pred.GetOrigin() == gt.GetOrigin()
        and pred.GetDirection() == gt.GetDirection()
    ):
        return to_np_zyx(pred)

    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(gt)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    rs.SetDefaultPixelValue(0)
    out = rs.Execute(pred)
    return to_np_zyx(out)


def dice_score(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    p_sum = int(np.count_nonzero(p))
    g_sum = int(np.count_nonzero(g))
    if p_sum == 0 and g_sum == 0:
        return 1.0
    if p_sum == 0 or g_sum == 0:
        return 0.0
    inter = int(np.count_nonzero(p & g))
    return float(2.0 * inter / (p_sum + g_sum))


def binary_surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        return mask
    eroded = binary_erosion(mask)
    if not np.any(eroded):
        return mask
    return mask & ~eroded


def crop_union_bbox(
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


def hd95_score(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing_zyx: Sequence[float]) -> Optional[float]:
    """ROI 裁剪 + 表面距离 EDT，比 medpy 全卷 HD95 快一个数量级以上。"""
    p = pred_bin.astype(bool, copy=False)
    g = gt_bin.astype(bool, copy=False)
    if not np.any(p) and not np.any(g):
        return 0.0
    if not np.any(p) or not np.any(g):
        return None

    p, g = crop_union_bbox(p, g, margin=3)
    p_surf = binary_surface(p)
    g_surf = binary_surface(g)
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


def safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return None
    return float(np.mean(vals))


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


def evaluate_one_case(
    gt_path: Path,
    pred_path: Path,
    label_items: List[Tuple[int, str]],
    compute_hd95: bool = True,
) -> Dict[str, Any]:
    gt_img = sitk.ReadImage(str(gt_path))
    pred_img = sitk.ReadImage(str(pred_path))
    gt_arr = to_np_zyx(gt_img)
    pred_arr = resample_pred_to_gt(pred_img, gt_img)
    if gt_arr.shape != pred_arr.shape:
        raise ValueError(f"shape不一致: {gt_path.name}, gt={gt_arr.shape}, pred={pred_arr.shape}")

    spacing = voxel_spacing_zyx(gt_img)
    case_metrics: Dict[str, Dict[str, Optional[float]]] = {}
    this_case_dice: List[Optional[float]] = []
    this_case_hd: List[Optional[float]] = []

    for lid, lname in label_items:
        pb = pred_arr == lid
        gb = gt_arr == lid
        d = dice_score(pb, gb)
        h = hd95_score(pb, gb, spacing) if compute_hd95 else None

        case_metrics[str(lid)] = {"name": lname, "Dice": d, "HD95": h}
        this_case_dice.append(d)
        this_case_hd.append(h)

    return {
        "case": gt_path.name,
        "metrics": case_metrics,
        "macro_dice": safe_mean(this_case_dice),
        "macro_hd95": safe_mean(this_case_hd) if compute_hd95 else None,
    }


def evaluate_one_pred_dir(
    pred_dir: Path,
    gt_dir: Path,
    label_items: List[Tuple[int, str]],
    workers: int = 1,
    compute_hd95: bool = True,
) -> Dict[str, Any]:
    if not pred_dir.exists():
        raise FileNotFoundError(f"预测目录不存在: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"GT目录不存在: {gt_dir}")

    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"GT目录下没有 .nii.gz 文件: {gt_dir}")

    per_case: List[Dict[str, Any]] = []

    if workers <= 1:
        iterator = gt_files
        if tqdm is not None:
            iterator = tqdm(gt_files, desc=f"{pred_dir.name}", unit="case")
        for gt_path in iterator:
            pred_path = pred_dir / gt_path.name
            if not pred_path.exists():
                raise FileNotFoundError(f"缺少预测文件: {pred_path}")
            per_case.append(
                evaluate_one_case(gt_path, pred_path, label_items, compute_hd95=compute_hd95)
            )
    else:
        tasks = []
        for gt_path in gt_files:
            pred_path = pred_dir / gt_path.name
            if not pred_path.exists():
                raise FileNotFoundError(f"缺少预测文件: {pred_path}")
            tasks.append((gt_path, pred_path))

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    evaluate_one_case,
                    gt_path,
                    pred_path,
                    label_items,
                    compute_hd95,
                ): gt_path.name
                for gt_path, pred_path in tasks
            }
            done_map: Dict[str, Dict[str, Any]] = {}
            iterator = as_completed(futures)
            if tqdm is not None:
                iterator = tqdm(iterator, total=len(futures), desc=f"{pred_dir.name}", unit="case")
            for fut in iterator:
                result = fut.result()
                done_map[result["case"]] = result
        per_case = [done_map[gt_path.name] for gt_path in gt_files]

    dice_by_label: Dict[int, List[Optional[float]]] = {lid: [] for lid, _ in label_items}
    hd_by_label: Dict[int, List[Optional[float]]] = {lid: [] for lid, _ in label_items}
    case_macro_dice: List[Optional[float]] = []
    case_macro_hd: List[Optional[float]] = []

    for case_result in per_case:
        for lid, _ in label_items:
            m = case_result["metrics"][str(lid)]
            dice_by_label[lid].append(m["Dice"])
            hd_by_label[lid].append(m["HD95"])
        case_macro_dice.append(case_result["macro_dice"])
        case_macro_hd.append(case_result["macro_hd95"])

    per_label_mean: Dict[str, Dict[str, Optional[float]]] = {}
    for lid, lname in label_items:
        per_label_mean[str(lid)] = {
            "name": lname,
            "Dice": safe_mean(dice_by_label[lid]),
            "HD95": safe_mean(hd_by_label[lid]) if compute_hd95 else None,
        }

    return {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "n_cases": len(per_case),
        "mean": per_label_mean,
        "foreground_mean": {
            "Dice": safe_mean(case_macro_dice),
            "HD95": safe_mean(case_macro_hd) if compute_hd95 else None,
        },
        "metric_per_case": per_case,
    }


def write_outputs(summary: Dict[str, Any], pred_dir: Path) -> None:
    json_path = pred_dir / METRIC_JSON
    csv_path = pred_dir / METRIC_CSV

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    rows: List[Dict[str, Any]] = []
    for lid, info in summary["mean"].items():
        rows.append(
            {
                "label_id": lid,
                "label_name": info.get("name", ""),
                "Dice": info.get("Dice", None),
                "HD95_mm": info.get("HD95", None),
            }
        )
    rows.append(
        {
            "label_id": "avg",
            "label_name": "foreground_mean",
            "Dice": summary["foreground_mean"].get("Dice", None),
            "HD95_mm": summary["foreground_mean"].get("HD95", None),
        }
    )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["label_id", "label_name", "Dice", "HD95_mm"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"已写入: {json_path}")
    print(f"已写入: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 Synapse 测试集 Dice/HD95（每类+平均）")
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="数据集根目录（默认从该目录推导 labelsTs 与 dataset.json）",
    )
    parser.add_argument("--gt", type=str, default=None, help="测试集GT目录（labelsTs），不传则用 <root>/labelsTs")
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="dataset.json 路径；不传时默认使用 <root>/dataset.json",
    )
    parser.add_argument(
        "--pred-dirs",
        type=str,
        nargs="+",
        default=None,
        help="一个或多个预测目录（提供后将覆盖自动发现）",
    )
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="自动扫描 <root>/labelsTs_*/{final_pre,best_pre}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使结果文件已存在也强制重算并覆盖",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 4))),
        help="并行评估 case 的进程数（默认 min(8, CPU核数)）",
    )
    parser.add_argument(
        "--dice-only",
        action="store_true",
        help="仅计算 Dice，跳过 HD95（最快）",
    )
    args = parser.parse_args()

    root = Path(args.root)
    gt_dir = Path(args.gt) if args.gt else (root / "labelsTs")
    dataset_json = Path(args.dataset_json) if args.dataset_json else (root / "dataset.json")
    if args.pred_dirs:
        pred_dirs = [Path(p) for p in args.pred_dirs]
    else:
        pred_dirs = discover_pred_dirs(root)
        if not pred_dirs:
            raise FileNotFoundError(
                f"在 root 下未找到 labelsTs_*/best_pre 或 labelsTs_*/final_pre: {root}\n"
                f"可改用 --pred-dirs 显式传入目录。"
            )

    label_items = load_label_map(dataset_json)
    if not label_items:
        raise RuntimeError("未从 dataset.json 解析到前景标签")

    print("评估标签:")
    for lid, lname in label_items:
        print(f"  {lid}: {lname}")
    print(f"并行 workers={args.workers}, HD95={'关闭' if args.dice_only else '开启(EDT+ROI)'}")

    pred_iter = pred_dirs
    if tqdm is not None:
        pred_iter = tqdm(pred_dirs, desc="pred_dirs", unit="dir")

    for pd in pred_iter:
        if outputs_exist(pd) and not args.force:
            print(f"\n跳过已评估目录: {pd}（已存在 {METRIC_JSON}/{METRIC_CSV}）")
            continue
        print(f"\n开始评估: {pd}")
        summary = evaluate_one_pred_dir(
            pd,
            gt_dir,
            label_items,
            workers=args.workers,
            compute_hd95=not args.dice_only,
        )
        write_outputs(summary, pd)
        fg = summary["foreground_mean"]
        print(
            f"[{pd.name}] ForegroundMean Dice={fg.get('Dice'):.4f} "
            f"HD95={fg.get('HD95'):.4f}" if fg.get("Dice") is not None and fg.get("HD95") is not None
            else f"[{pd.name}] ForegroundMean Dice={fg.get('Dice')} HD95={fg.get('HD95')}"
        )


if __name__ == "__main__":
    main()
