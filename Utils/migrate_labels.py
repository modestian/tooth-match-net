"""
migrate_labels.py
=================
将旧版 labels.csv（只有 sample_id, label 两列）迁移为新版（含 split 列）。

用法：
    cd ToothMatchNet
    python Utils/migrate_labels.py

逻辑：
  1. 读取 MatchingData/labels.csv（旧格式）
  2. 扫描 MatchingData/train/ val/ test/ 三个目录下的子目录名
  3. 根据子目录所在位置确定 split
  4. 写出新的 labels.csv（sample_id, split, label）

如果某个 sample_id 在旧 CSV 中存在、但在磁盘上找不到对应子目录，会打印警告并跳过。
如果某个磁盘子目录在旧 CSV 中没有 label 记录，会打印警告（label 留空，需手动填写）。
"""

import csv
from pathlib import Path

# 从 Utils/ 目录向上两级到达项目根目录，再进入 MatchingData
BASE      = Path(__file__).parent.parent / "MatchingData"
LABELS_IN = BASE / "labels.csv"
SPLITS    = ["train", "val", "test"]


def main():
    # ---- 读取旧 CSV ----
    old_records: dict[str, str] = {}   # sample_id → label
    with open(LABELS_IN, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {(k.strip().lstrip("\ufeff") if k else k): v for k, v in row.items()}
            if all(v is None or (v and v.strip() == "") for v in row.values()):
                continue
            sid = row.get("sample_id", "").strip()
            lbl = row.get("label", "").strip()

            # 如果已经有 split 列，直接跳过迁移
            if "split" in row:
                print("[migrate] labels.csv already has a 'split' column — no migration needed.")
                return

            if sid:
                old_records[sid] = lbl

    print(f"[migrate] Read {len(old_records)} records from old labels.csv")

    # ---- 扫描磁盘目录 ----
    disk_map: dict[str, str] = {}   # sample_id → split
    for split in SPLITS:
        split_dir = BASE / split
        if not split_dir.is_dir():
            print(f"[migrate] WARNING: split directory not found: {split_dir}")
            continue
        for sub in sorted(split_dir.iterdir()):
            if sub.is_dir():
                if sub.name in disk_map:
                    print(f"[migrate] WARNING: '{sub.name}' found in multiple splits "
                          f"({disk_map[sub.name]} and {split}) — keeping {disk_map[sub.name]}")
                else:
                    disk_map[sub.name] = split

    print(f"[migrate] Found {len(disk_map)} sample directories on disk "
          f"({', '.join(f'{s}: {sum(v==s for v in disk_map.values())}' for s in SPLITS)})")

    # ---- 合并 ----
    rows_out = []
    skipped_no_disk  = []
    skipped_no_label = []

    # samples present in old CSV
    for sid, lbl in sorted(old_records.items()):
        if sid not in disk_map:
            skipped_no_disk.append(sid)
            continue
        rows_out.append({"sample_id": sid, "split": disk_map[sid], "label": lbl})

    # samples on disk but missing from old CSV
    for sid, split in sorted(disk_map.items()):
        if sid not in old_records:
            skipped_no_label.append(sid)
            rows_out.append({"sample_id": sid, "split": split, "label": ""})

    if skipped_no_disk:
        print(f"\n[migrate] WARNING: {len(skipped_no_disk)} sample(s) in CSV but "
              f"not on disk (skipped):")
        for s in skipped_no_disk:
            print(f"    {s}")

    if skipped_no_label:
        print(f"\n[migrate] WARNING: {len(skipped_no_label)} sample(s) on disk but "
              f"not in old CSV (label left empty — fill manually):")
        for s in skipped_no_label:
            print(f"    {s}")

    # ---- 写出新 CSV ----
    # Sort: train first, then val, then test; within each split by sample_id
    split_order = {s: i for i, s in enumerate(SPLITS)}
    rows_out.sort(key=lambda r: (split_order.get(r["split"], 99), r["sample_id"]))

    with open(LABELS_IN, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "split", "label"])
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"\n[migrate] ✓ Wrote {len(rows_out)} rows to {LABELS_IN}")
    print("          Columns: sample_id, split, label")
    if skipped_no_label:
        print("          ⚠ Fill in empty 'label' values before training!")


if __name__ == "__main__":
    main()
