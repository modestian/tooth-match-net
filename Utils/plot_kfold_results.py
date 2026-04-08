"""
plot_kfold_results.py
=====================
将 K-Fold 训练日志转换为可视化图表。

用法：
    cd ToothMatchNet
    python Utils/plot_kfold_results.py

功能：
  1. 读取 MatchingCheckpoints/KFold/ 目录下所有 fold_* 文件夹中的 train.log
  2. 解析日志提取训练曲线数据（Loss、Accuracy、F1、Learning Rate）
  3. 生成以下图表：
     - 各折训练曲线对比图
     - 各折最佳性能柱状图
     - 汇总性能统计图
  4. 输出到 Report/KFold/ 目录

输出文件：
  - training_curves_all_folds.png    各折训练曲线对比
  - best_f1_comparison.png           各折最佳F1对比柱状图
  - kfold_summary_report.md          Markdown汇总报告
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
KFOLD_DIR = PROJECT_ROOT / "MatchingCheckpoints" / "KFold"
OUTPUT_DIR = PROJECT_ROOT / "Report" / "KFold"


def parse_fold_log(log_path: Path) -> Dict:
    """
    解析单个 fold 的 train.log 文件。
    
    返回:
        {
            'fold': int,
            'epochs': List[int],
            'train_loss': List[float],
            'train_acc': List[float],
            'val_loss': List[float],
            'val_acc': List[float],
            'val_f1': List[float],
            'val_precision': List[float],
            'val_recall': List[float],
            'lr': List[float],
            'best_f1': float,
            'best_epoch': int,
        }
    """
    data = {
        'fold': None,
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': [],
        'lr': [],
        'best_f1': 0.0,
        'best_epoch': 0,
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取 fold 编号
    fold_match = re.search(r'Fold (\d+)/\d+', content)
    if fold_match:
        data['fold'] = int(fold_match.group(1))
    
    # 提取每个 epoch 的数据
    epoch_pattern = re.compile(
        r'Epoch (\d+)/\d+\s*\|\s*'
        r'LR=([\d.e+-]+)\s*\|\s*'
        r'Train Loss=([\d.]+)\s+Acc=([\d.]+)\s*\|\s*'
        r'Val Loss=([\d.]+)\s+Acc=([\d.]+)\s+F1=([\d.]+)\s*\|\s*'
        r'P=([\d.]+)\s+R=([\d.]+)'
    )
    
    for match in epoch_pattern.finditer(content):
        epoch = int(match.group(1))
        lr = float(match.group(2))
        train_loss = float(match.group(3))
        train_acc = float(match.group(4))
        val_loss = float(match.group(5))
        val_acc = float(match.group(6))
        val_f1 = float(match.group(7))
        val_precision = float(match.group(8))
        val_recall = float(match.group(9))
        
        data['epochs'].append(epoch)
        data['lr'].append(lr)
        data['train_loss'].append(train_loss)
        data['train_acc'].append(train_acc)
        data['val_loss'].append(val_loss)
        data['val_acc'].append(val_acc)
        data['val_f1'].append(val_f1)
        data['val_precision'].append(val_precision)
        data['val_recall'].append(val_recall)
        
        # 更新最佳 F1
        if val_f1 > data['best_f1']:
            data['best_f1'] = val_f1
            data['best_epoch'] = epoch
    
    return data


def load_all_folds(kfold_dir: Path) -> List[Dict]:
    """加载所有 fold 的训练数据"""
    all_folds = []
    
    if not kfold_dir.exists():
        print(f"[ERROR] KFold directory not found: {kfold_dir}")
        return all_folds
    
    # 查找所有 fold_* 目录
    fold_dirs = sorted(kfold_dir.glob("fold_*"))
    
    for fold_dir in fold_dirs:
        log_path = fold_dir / "train.log"
        if log_path.exists():
            print(f"[INFO] Parsing {log_path}")
            fold_data = parse_fold_log(log_path)
            if fold_data['epochs']:
                all_folds.append(fold_data)
        else:
            print(f"[WARNING] Log file not found: {log_path}")
    
    return all_folds


def plot_training_curves_all_folds(all_folds: List[Dict], output_path: Path):
    """绘制所有 fold 的训练曲线对比图"""
    if not all_folds:
        print("[WARNING] No fold data to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-Fold Training Curves Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_folds)))
    
    # Loss 曲线
    ax = axes[0, 0]
    for i, fold in enumerate(all_folds):
        ax.plot(fold['epochs'], fold['train_loss'], '-', color=colors[i], 
                label=f"Fold {fold['fold']} Train", alpha=0.7)
        ax.plot(fold['epochs'], fold['val_loss'], '--', color=colors[i],
                label=f"Fold {fold['fold']} Val", alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    ax = axes[0, 1]
    for i, fold in enumerate(all_folds):
        ax.plot(fold['epochs'], fold['train_acc'], '-', color=colors[i], alpha=0.7)
        ax.plot(fold['epochs'], fold['val_acc'], '--', color=colors[i], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves (solid=train, dashed=val)')
    ax.grid(True, alpha=0.3)
    
    # F1 Score 曲线
    ax = axes[1, 0]
    for i, fold in enumerate(all_folds):
        ax.plot(fold['epochs'], fold['val_f1'], '-', color=colors[i],
                label=f"Fold {fold['fold']}", alpha=0.7)
        ax.axhline(y=fold['best_f1'], color=colors[i], linestyle=':', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score (dotted=best)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Learning Rate 曲线
    ax = axes[1, 1]
    for i, fold in enumerate(all_folds):
        ax.plot(fold['epochs'], fold['lr'], '-', color=colors[i], alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_best_f1_comparison(all_folds: List[Dict], output_path: Path):
    """绘制各折最佳 F1 对比柱状图"""
    if not all_folds:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fold_nums = [f"Fold {f['fold']}" for f in all_folds]
    best_f1s = [f['best_f1'] for f in all_folds]
    best_epochs = [f['best_epoch'] for f in all_folds]
    
    # 计算平均值和标准差
    avg_f1 = np.mean(best_f1s)
    std_f1 = np.std(best_f1s)
    
    # 绘制柱状图
    colors = ['#2ecc71' if f1 >= avg_f1 else '#e74c3c' for f1 in best_f1s]
    bars = ax.bar(fold_nums, best_f1s, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, f1, epoch in zip(bars, best_f1s, best_epochs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{f1:.4f}\n(Epoch {epoch})',
                ha='center', va='bottom', fontsize=9)
    
    # 添加平均线
    ax.axhline(y=avg_f1, color='blue', linestyle='--', linewidth=2,
               label=f'Average: {avg_f1:.4f} ± {std_f1:.4f}')
    
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Best F1 Score', fontsize=12)
    ax.set_title('K-Fold Best F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, max(best_f1s) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def plot_metrics_summary(all_folds: List[Dict], output_path: Path):
    """绘制各折指标汇总图"""
    if not all_folds:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('K-Fold Metrics Summary', fontsize=14, fontweight='bold')
    
    fold_nums = [f['fold'] for f in all_folds]
    
    # Precision vs Recall
    ax = axes[0]
    precisions = [f['val_precision'][f['epochs'].index(f['best_epoch'])] if f['best_epoch'] in f['epochs'] else 0 for f in all_folds]
    recalls = [f['val_recall'][f['epochs'].index(f['best_epoch'])] if f['best_epoch'] in f['epochs'] else 0 for f in all_folds]
    
    x = np.arange(len(fold_nums))
    width = 0.35
    
    ax.bar(x - width/2, precisions, width, label='Precision', color='#3498db')
    ax.bar(x + width/2, recalls, width, label='Recall', color='#e74c3c')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_title('Precision vs Recall at Best Epoch')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in fold_nums])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Training Time (epochs)
    ax = axes[1]
    total_epochs = [len(f['epochs']) for f in all_folds]
    best_epochs = [f['best_epoch'] for f in all_folds]
    
    ax.bar(x - width/2, total_epochs, width, label='Total Epochs', color='#9b59b6')
    ax.bar(x + width/2, best_epochs, width, label='Best Epoch', color='#2ecc71')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Epoch')
    ax.set_title('Training Progress')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in fold_nums])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # F1 Distribution
    ax = axes[2]
    best_f1s = [f['best_f1'] for f in all_folds]
    avg_f1 = np.mean(best_f1s)
    std_f1 = np.std(best_f1s)
    
    ax.bar(fold_nums, best_f1s, color='#3498db', edgecolor='black')
    ax.axhline(y=avg_f1, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_f1:.4f}')
    ax.fill_between([min(fold_nums)-0.5, max(fold_nums)+0.5], 
                    avg_f1 - std_f1, avg_f1 + std_f1, 
                    alpha=0.2, color='red', label=f'±1 Std: {std_f1:.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Best F1 Score')
    ax.set_title('F1 Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved: {output_path}")


def generate_markdown_report(all_folds: List[Dict], output_path: Path):
    """生成 Markdown 汇总报告"""
    if not all_folds:
        return
    
    # 尝试读取 kfold_summary.json
    summary_path = KFOLD_DIR / "kfold_summary.json"
    config = {}
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            summary_json = json.load(f)
            config = summary_json.get('config', {})
    
    avg_f1 = np.mean([f['best_f1'] for f in all_folds])
    std_f1 = np.std([f['best_f1'] for f in all_folds])
    
    report = f"""# K-Fold Cross Validation Report

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 训练配置

| 参数 | 值 |
|------|-----|
| Backbone | {config.get('backbone', 'N/A')} |
| Epochs | {config.get('epochs', 'N/A')} |
| Batch Size | {config.get('batch_size', 'N/A')} |
| Learning Rate | {config.get('learning_rate', 'N/A')} |
| K-Folds | {config.get('k_folds', len(all_folds))} |

---

## 2. 各折结果汇总

| Fold | Best F1 | Best Epoch | Total Epochs | Precision | Recall |
|------|---------|------------|--------------|-----------|--------|
"""
    
    for fold in all_folds:
        best_idx = fold['epochs'].index(fold['best_epoch']) if fold['best_epoch'] in fold['epochs'] else -1
        precision = fold['val_precision'][best_idx] if best_idx >= 0 else 0
        recall = fold['val_recall'][best_idx] if best_idx >= 0 else 0
        report += f"| {fold['fold']} | {fold['best_f1']:.4f} | {fold['best_epoch']} | {len(fold['epochs'])} | {precision:.4f} | {recall:.4f} |\n"
    
    report += f"""
---

## 3. 统计汇总

| 指标 | 值 |
|------|-----|
| **Average F1** | {avg_f1:.4f} |
| **Std F1** | {std_f1:.4f} |
| **Best Fold** | Fold {max(all_folds, key=lambda x: x['best_f1'])['fold']} ({max(f['best_f1'] for f in all_folds):.4f}) |
| **Worst Fold** | Fold {min(all_folds, key=lambda x: x['best_f1'])['fold']} ({min(f['best_f1'] for f in all_folds):.4f}) |

---

## 4. 可视化图表

### 4.1 训练曲线对比

![训练曲线](training_curves_all_folds.png)

### 4.2 最佳F1对比

![F1对比](best_f1_comparison.png)

### 4.3 指标汇总

![指标汇总](metrics_summary.png)

---

## 5. 结论

K-Fold 交叉验证结果显示：

- 平均 F1 Score 为 **{avg_f1:.4f} ± {std_f1:.4f}**
- 各折之间存在{'较大' if std_f1 > 0.05 else '较小'}波动（标准差 = {std_f1:.4f}）
- 最佳折达到 F1 = {max(f['best_f1'] for f in all_folds):.4f}

### 建议

"""
    
    if std_f1 > 0.1:
        report += "- 各折差异较大，建议检查数据分布是否均衡\n"
    if avg_f1 < 0.7:
        report += "- 平均 F1 较低，建议增加训练数据或调整超参数\n"
    if avg_f1 > 0.8:
        report += "- 模型表现良好，可以考虑在测试集上验证\n"
    
    report += """
---

*本报告由 plot_kfold_results.py 自动生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"[INFO] Saved: {output_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("K-Fold Results Visualization Tool")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载所有 fold 数据
    print(f"\n[INFO] Loading data from: {KFOLD_DIR}")
    all_folds = load_all_folds(KFOLD_DIR)
    
    if not all_folds:
        print("[ERROR] No fold data found. Please run kfold_train.py first.")
        return
    
    print(f"[INFO] Loaded {len(all_folds)} folds")
    
    # 生成图表
    print("\n[INFO] Generating plots...")
    
    plot_training_curves_all_folds(
        all_folds, 
        OUTPUT_DIR / "training_curves_all_folds.png"
    )
    
    plot_best_f1_comparison(
        all_folds,
        OUTPUT_DIR / "best_f1_comparison.png"
    )
    
    plot_metrics_summary(
        all_folds,
        OUTPUT_DIR / "metrics_summary.png"
    )
    
    # 生成报告
    print("\n[INFO] Generating report...")
    generate_markdown_report(
        all_folds,
        OUTPUT_DIR / "kfold_summary_report.md"
    )
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
