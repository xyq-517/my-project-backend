import csv
import os
from os.path import join

import numpy as np
from PIL import Image
from tqdm import tqdm

from medpy.metric.binary import hd95 as hd95_metric

from utils.utils_metrics import fast_hist, per_class_iu, per_class_PA_Recall, per_class_Precision, per_Accuracy


def dice_from_hist(hist):
    tp = np.diag(hist).astype(np.float64)
    fp = hist.sum(axis=0).astype(np.float64) - tp
    fn = hist.sum(axis=1).astype(np.float64) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, np.maximum(denom, 1.0), dtype=np.float64)


def hd95_per_class(label, pred, num_classes, empty_penalty=50.0):
    scores = np.full((num_classes,), np.nan, dtype=np.float64)
    for k in range(num_classes):
        gt_k = (label == k)
        pr_k = (pred == k)
        has_gt = bool(np.any(gt_k))
        has_pr = bool(np.any(pr_k))
        if has_gt and has_pr:
            scores[k] = float(hd95_metric(pr_k, gt_k))
        elif (not has_gt) and (not has_pr):
            scores[k] = np.nan
        else:
            scores[k] = float(empty_penalty)
    return scores


def nanmean_or_nan(x):
    if np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


if __name__ == "__main__":
    dataset = "LIDC"  # "kits19" or "LIDC"
    split = "test"
    
    if dataset == "kits19":
        num_classes = 3
        name_classes = ["background", "kidney", "tumor"]
        VOCdevkit_path = r"D:\shengwu\using\unet-pytorch\VOCdevkit_kits19"
    else:
        num_classes = 2
        name_classes = ["background", "nodule"]
        VOCdevkit_path = r"D:\shengwu\using\VOCdevkit_lidc"
    
    miou_out_path = "miou_out"
    include_background_in_mean = False
    empty_penalty_hd95 = 50.0

    image_ids = open(join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/" + split + ".txt"), "r").read().splitlines()
    img_dir = join(VOCdevkit_path, "VOC2007/JPEGImages")  # 原始图目录
    gt_dir = join(VOCdevkit_path, "VOC2007/SegmentationClass")
    pred_dir = join(miou_out_path, "detection-results")

    if not os.path.isdir(pred_dir):
        raise FileNotFoundError("Prediction directory not found: " + pred_dir)

    out_csv = join(miou_out_path, "per_image_metrics_" + dataset + "_" + split + ".csv")
    class_indices_for_mean = list(range(num_classes)) if include_background_in_mean else list(range(1, num_classes))

    print("=" * 70)
    print("Processing dataset:", dataset)
    print("Number of images:", len(image_ids))
    print("=" * 70)

    rows = []
    matched_count = 0
    unmatched_count = 0
    
    for image_id in image_ids:
        img_path = join(img_dir, image_id + ".jpg")    # 原始图
        gt_path = join(gt_dir, image_id + ".png")      # 金标准
        pred_path = join(pred_dir, image_id + ".png")  # 分割结果
        
        if not os.path.exists(img_path):
            print(f"[缺失] 原始图: {image_id}.jpg")
            unmatched_count += 1
            continue
        if not os.path.exists(gt_path):
            print(f"[缺失] 金标准: {image_id}.png")
            unmatched_count += 1
            continue
        if not os.path.exists(pred_path):
            print(f"[缺失] 分割图: {image_id}.png")
            unmatched_count += 1
            continue

        # 输出三张图的名称
        print("\n[匹配成功]")
        print("  原始图:", image_id + ".jpg")
        print("  分割图:", image_id + ".png")
        print("  金标准:", image_id + ".png")

        label = np.array(Image.open(gt_path), dtype=np.int64)
        pred = np.array(Image.open(pred_path), dtype=np.int64)
        
        if label.shape != pred.shape:
            print(f"[错误] 尺寸不匹配: {image_id}")
            unmatched_count += 1
            continue

        hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
        ious = per_class_iu(hist)
        recalls = per_class_PA_Recall(hist)
        precisions = per_class_Precision(hist)
        dices = dice_from_hist(hist)
        hd95s = hd95_per_class(label, pred, num_classes, empty_penalty=empty_penalty_hd95)

        mean_iou = nanmean_or_nan(ious[class_indices_for_mean])
        mean_dice = nanmean_or_nan(dices[class_indices_for_mean])
        mean_recall = nanmean_or_nan(recalls[class_indices_for_mean])
        mean_precision = nanmean_or_nan(precisions[class_indices_for_mean])
        mean_hd95 = nanmean_or_nan(hd95s[class_indices_for_mean])
        acc = float(per_Accuracy(hist))

        row = {
            "image_id": image_id,
            "accuracy": acc,
            "mIoU": mean_iou,
            "mDice": mean_dice,
            "mRecall": mean_recall,
            "mPrecision": mean_precision,
            "mHD95": mean_hd95,
        }
        for k in range(num_classes):
            name = name_classes[k] if k < len(name_classes) else "class_" + str(k)
            row["IoU_" + name] = float(ious[k]) if not np.isnan(ious[k]) else float("nan")
            row["Dice_" + name] = float(dices[k]) if not np.isnan(dices[k]) else float("nan")
            row["HD95_" + name] = float(hd95s[k]) if not np.isnan(hd95s[k]) else float("nan")
        rows.append(row)
        matched_count += 1

    print("\n" + "=" * 70)
    print(f"匹配成功: {matched_count} 张")
    print(f"匹配失败: {unmatched_count} 张")
    print("=" * 70)

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        avg = {}
        for key in ["accuracy", "mIoU", "mDice", "mRecall", "mPrecision", "mHD95"]:
            avg[key] = nanmean_or_nan(np.array([r[key] for r in rows], dtype=np.float64))
        print("Saved:", out_csv)
        print("Average metrics:")
        for k, v in avg.items():
            print("  " + k + ": " + str(round(v, 4)))