"""
split_data_ordered.py — 按顺序为 ToothMatchNet 生成 train/val/test 数据划分
规则：前 70% 放入 train，剩余 30% 平均分配给 val 和 test (各 15%)
"""

import csv
from pathlib import Path


def main():
    total_no_match = 279  # sample_0001 ~ sample_0279
    total_match = 491     # sample_1001 ~ sample_1491

    print(f"总样本数：{total_match + total_no_match}")
    print(f"  匹配样本：{total_match}")
    print(f"  不匹配样本：{total_no_match}")
    print()

    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    # 不匹配样本分配 (0001-0279)
    no_match_train_end = int(total_no_match * train_ratio)  # 前 70%
    no_match_val_end = int(total_no_match * (train_ratio + val_ratio))  # 前 85%

    no_match_train = list(range(1, no_match_train_end + 1))
    no_match_val = list(range(no_match_train_end + 1, no_match_val_end + 1))
    no_match_test = list(range(no_match_val_end + 1, total_no_match + 1))

    # 匹配样本分配 (1001-1491)
    match_train_end = int(total_match * train_ratio)  # 前 70%
    match_val_end = int(total_match * (train_ratio + val_ratio))  # 前 85%

    match_train = list(range(1001, 1001 + match_train_end))
    match_val = list(range(1001 + match_train_end, 1001 + match_val_end))
    match_test = list(range(1001 + match_val_end, 1001 + total_match))

    # 生成 CSV
    samples = []

    # 先不匹配样本 (0001-0279)
    for idx in no_match_train:
        samples.append({"sample_id": f"sample_{idx:04d}", "split": "train", "label": 0})
    for idx in no_match_val:
        samples.append({"sample_id": f"sample_{idx:04d}", "split": "val", "label": 0})
    for idx in no_match_test:
        samples.append({"sample_id": f"sample_{idx:04d}", "split": "test", "label": 0})

    # 后匹配样本 (1001-1491)
    for idx in match_train:
        samples.append({"sample_id": f"sample_{idx:04d}", "split": "train", "label": 1})
    for idx in match_val:
        samples.append({"sample_id": f"sample_{idx:04d}", "split": "val", "label": 1})
    for idx in match_test:
        samples.append({"sample_id": f"sample_{idx:04d}", "split": "test", "label": 1})

    # 统计
    train_count = sum(1 for s in samples if s["split"] == "train")
    val_count = sum(1 for s in samples if s["split"] == "val")
    test_count = sum(1 for s in samples if s["split"] == "test")

    train_pos = sum(1 for s in samples if s["split"] == "train" and s["label"] == 1)
    train_neg = sum(1 for s in samples if s["split"] == "train" and s["label"] == 0)
    val_pos = sum(1 for s in samples if s["split"] == "val" and s["label"] == 1)
    val_neg = sum(1 for s in samples if s["split"] == "val" and s["label"] == 0)
    test_pos = sum(1 for s in samples if s["split"] == "test" and s["label"] == 1)
    test_neg = sum(1 for s in samples if s["split"] == "test" and s["label"] == 0)

    print("数据分配结果:")
    print("=" * 60)
    print(f"不匹配样本 (0001-0279):")
    print(f"  Train: {len(no_match_train)} (0001-{no_match_train_end:04d})")
    print(f"  Val:   {len(no_match_val)} ({no_match_train_end+1:04d}-{no_match_val_end:04d})")
    print(f"  Test:  {len(no_match_test)} ({no_match_val_end+1:04d}-{total_no_match:04d})")
    print()
    print(f"匹配样本 (1001-1491):")
    print(f"  Train: {len(match_train)} (1001-{1001+match_train_end-1:04d})")
    print(f"  Val:   {len(match_val)} ({1001+match_train_end:04d}-{1001+match_val_end-1:04d})")
    print(f"  Test:  {len(match_test)} ({1001+match_val_end:04d}-{1001+total_match-1:04d})")
    print()
    print("汇总:")
    print("=" * 60)
    print(f"Train: {train_count} samples (pos={train_pos}, neg={train_neg})")
    print(f"Val:   {val_count} samples (pos={val_pos}, neg={val_neg})")
    print(f"Test:  {test_count} samples (pos={test_pos}, neg={test_neg})")
    print("=" * 60)

    # 写入文件
    output_path = Path(__file__).parent.parent / "MatchingData" / "labels.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "label"])
        writer.writeheader()
        for s in samples:
            writer.writerow(s)

    print(f"\n新 labels.csv 已保存至：{output_path}")


if __name__ == "__main__":
    main()
