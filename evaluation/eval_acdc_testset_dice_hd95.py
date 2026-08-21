"""
ACDC 测试集评估脚本：Dice + HD95（每类 + 平均）

默认目录：
- root: D:\zhuomian\data\nnUNet_raw\Dataset100_ACDC
  - gt: <root>\labelsTs
  - dataset.json: <root>\dataset.json
  - 预测目录自动发现：<root>\labelsTs_*\final_pre 或 best_pre

输出（每个预测目录）：
- metrics_ACDC_testset.json
- metrics_ACDC_testset.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import SimpleITK as sitk
except ImportError as e:
    raise ImportError("需要 SimpleITK，请先安装: pip install SimpleITK") from e

try:
    from medpy.metric.binary import hd95 as medpy_hd95
except ImportError as e:
    raise ImportError("需要 medpy，请先安装: pip install medpy") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore


DEFAULT_ROOT = Path(r"D:\zhuomian\data\nnUNet_raw\Dataset100_ACDC")
DEFAULT_GT_DIR = DEFAULT_ROOT / "labelsTs"
DEFAULT_DATASET_JSON = DEFAULT_ROOT / "dataset.json"


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


def hd95_score(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing_zyx: Sequence[float]) -> Optional[float]:
    p = pred_bin.astype(bool)
    g = gt_bin.astype(bool)
    if not np.any(p) and not np.any(g):
        return 0.0
    if not np.any(p) or not np.any(g):
        return None
    try:
        return float(medpy_hd95(p, g, voxelspacing=spacing_zyx))
    except Exception:
        return None


def safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return None
    return float(np.mean(vals))


def outputs_exist(pred_dir: Path) -> bool:
    return (pred_dir / "metrics_ACDC_testset.json").exists() and (pred_dir / "metrics_ACDC_testset.csv").exists()


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


def evaluate_one_pred_dir(pred_dir: Path, gt_dir: Path, label_items: List[Tuple[int, str]]) -> Dict[str, Any]:
    gt_files = sorted(gt_dir.glob("*.nii.gz"))
    if not gt_files:
        raise RuntimeError(f"GT目录下没有 .nii.gz 文件: {gt_dir}")

    per_case: List[Dict[str, Any]] = []
    dice_by_label: Dict[int, List[Optional[float]]] = {lid: [] for lid, _ in label_items}
    hd_by_label: Dict[int, List[Optional[float]]] = {lid: [] for lid, _ in label_items}
    case_macro_dice: List[Optional[float]] = []
    case_macro_hd: List[Optional[float]] = []

    iterator = gt_files
    if tqdm is not None:
        iterator = tqdm(gt_files, desc=pred_dir.parent.name + "/" + pred_dir.name, unit="case")

    for gt_path in iterator:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            raise FileNotFoundError(f"缺少预测文件: {pred_path}")

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
            h = hd95_score(pb, gb, spacing)

            case_metrics[str(lid)] = {"name": lname, "Dice": d, "HD95": h}
            dice_by_label[lid].append(d)
            hd_by_label[lid].append(h)
            this_case_dice.append(d)
            this_case_hd.append(h)

        per_case.append(
            {
                "case": gt_path.name,
                "metrics": case_metrics,
                "macro_dice": safe_mean(this_case_dice),
                "macro_hd95": safe_mean(this_case_hd),
            }
        )
        case_macro_dice.append(safe_mean(this_case_dice))
        case_macro_hd.append(safe_mean(this_case_hd))

    per_label_mean: Dict[str, Dict[str, Optional[float]]] = {}
    for lid, lname in label_items:
        per_label_mean[str(lid)] = {
            "name": lname,
            "Dice": safe_mean(dice_by_label[lid]),
            "HD95": safe_mean(hd_by_label[lid]),
        }

    return {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "n_cases": len(per_case),
        "mean": per_label_mean,
        "foreground_mean": {
            "Dice": safe_mean(case_macro_dice),
            "HD95": safe_mean(case_macro_hd),
        },
        "metric_per_case": per_case,
    }


def write_outputs(summary: Dict[str, Any], pred_dir: Path) -> None:
    out_json = pred_dir / "metrics_ACDC_testset.json"
    out_csv = pred_dir / "metrics_ACDC_testset.csv"

    with open(out_json, "w", encoding="utf-8") as f:
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

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["label_id", "label_name", "Dice", "HD95_mm"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"已写入: {out_json}")
    print(f"已写入: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 ACDC 测试集 Dice/HD95（每类+平均）")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT), help="数据集根目录")
    parser.add_argument("--gt", type=str, default=str(DEFAULT_GT_DIR), help="测试集GT目录")
    parser.add_argument("--dataset-json", type=str, default=str(DEFAULT_DATASET_JSON), help="dataset.json 路径")
    parser.add_argument("--pred-dirs", type=str, nargs="+", default=None, help="一个或多个预测目录")
    parser.add_argument(
        "--auto-discover",
        action="store_true",
        help="自动扫描 <root>/labelsTs_*/{final_pre,best_pre}",
    )
    parser.add_argument("--force", action="store_true", help="即使结果已存在也强制重算")
    args = parser.parse_args()

    root = Path(args.root)
    gt_dir = Path(args.gt)
    dataset_json = Path(args.dataset_json)

    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT目录不存在: {gt_dir}")
    if not dataset_json.is_file():
        raise FileNotFoundError(f"dataset.json 不存在: {dataset_json}")

    if args.pred_dirs:
        pred_dirs = [Path(p) for p in args.pred_dirs]
    elif args.auto_discover:
        pred_dirs = discover_pred_dirs(root)
    else:
        pred_dirs = discover_pred_dirs(root)

    if not pred_dirs:
        raise RuntimeError("未发现可评估的预测目录。可用 --pred-dirs 显式传入。")

    label_items = load_label_map(dataset_json)
    if not label_items:
        raise RuntimeError("未从 dataset.json 解析到前景标签")

    print("评估标签:")
    for lid, lname in label_items:
        print(f"  {lid}: {lname}")

    pred_iter = pred_dirs
    if tqdm is not None:
        pred_iter = tqdm(pred_dirs, desc="pred_dirs", unit="dir")

    for pd in pred_iter:
        if outputs_exist(pd) and not args.force:
            print(f"\n跳过已评估目录: {pd}（已存在 metrics_ACDC_testset.json/csv）")
            continue
        print(f"\n开始评估: {pd}")
        summary = evaluate_one_pred_dir(pd, gt_dir, label_items)
        write_outputs(summary, pd)
        fg = summary["foreground_mean"]
        print(
            f"[{pd.parent.name}/{pd.name}] ForegroundMean Dice={fg.get('Dice'):.4f} HD95={fg.get('HD95'):.4f}"
            if fg.get("Dice") is not None and fg.get("HD95") is not None
            else f"[{pd.parent.name}/{pd.name}] ForegroundMean Dice={fg.get('Dice')} HD95={fg.get('HD95')}"
        )


if __name__ == "__main__":
    main()

