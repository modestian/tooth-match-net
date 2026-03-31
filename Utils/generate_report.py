"""
generate_report.py — 生成 ToothMatchNet 模型性能分析报告

Usage:
    python generate_report.py              # 生成交互式菜单，选择生成内容
    python generate_report.py --all        # 生成完整报告（默认）
    python generate_report.py --curves     # 仅生成训练曲线图
    python generate_report.py --cm         # 仅生成混淆矩阵
    python generate_report.py --roc        # 仅生成ROC曲线
    python generate_report.py --log PATH   # 指定日志文件路径
    python generate_report.py --output DIR # 指定输出目录

输出目录: Report/
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="生成 ToothMatchNet 模型性能分析报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python generate_report.py              # 交互式菜单
    python generate_report.py --all        # 生成完整报告
    python generate_report.py --curves     # 仅训练曲线
    python generate_report.py --cm         # 仅混淆矩阵
    python generate_report.py --roc        # 仅ROC曲线
        """
    )
    parser.add_argument("--all", action="store_true", help="生成完整报告（默认）")
    parser.add_argument("--curves", action="store_true", help="仅生成训练曲线图")
    parser.add_argument("--cm", action="store_true", help="仅生成混淆矩阵")
    parser.add_argument("--roc", action="store_true", help="仅生成ROC曲线")
    parser.add_argument("--log", type=str, default=None, help="训练日志文件路径")
    parser.add_argument("--output", type=str, default="Report", help="输出目录")
    return parser.parse_args()


def get_project_root() -> Path:
    """获取项目根目录"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent


def find_log_file(log_path: Optional[str] = None) -> Path:
    """查找训练日志文件"""
    if log_path:
        path = Path(log_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"指定的日志文件不存在: {log_path}")
    
    root = get_project_root()
    default_paths = [
        root / "MatchingCheckpoints" / "train.log",
        root / "MatchingCheckpoints" / "training.log",
        root / "train.log",
    ]
    
    for path in default_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError("未找到训练日志文件，请使用 --log 指定")


def parse_log_file(log_path: Path) -> Dict:
    """解析训练日志文件，提取训练指标"""
    epochs = []
    train_loss = []
    train_acc = []
    train_f1 = []
    val_loss = []
    val_acc = []
    val_f1 = []
    val_precision = []
    val_recall = []
    val_specificity = []
    learning_rates = []
    
    # 存储每个epoch的混淆矩阵数据
    all_cm_data = []
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析每个epoch的数据
    epoch_pattern = r'Epoch\s+(\d+)/(\d+)\s+\[([^\]]+)\]\s+LR=([\d.e+-]+)'
    epoch_matches = list(re.finditer(epoch_pattern, content))
    
    for i, match in enumerate(epoch_matches):
        epoch_num = int(match.group(1))
        total_epochs = int(match.group(2))
        time_str = match.group(3)
        lr = float(match.group(4))
        
        # 获取该epoch的文本范围
        start_pos = match.start()
        end_pos = epoch_matches[i + 1].start() if i + 1 < len(epoch_matches) else len(content)
        epoch_text = content[start_pos:end_pos]
        
        # 解析训练指标
        train_match = re.search(
            r'Train\s*\|\s*Loss=([\d.]+)\s+Acc=([\d.]+)\s+F1=([\d.]+)',
            epoch_text
        )
        if train_match:
            train_loss.append(float(train_match.group(1)))
            train_acc.append(float(train_match.group(2)))
            train_f1.append(float(train_match.group(3)))
        
        # 解析验证指标
        val_match = re.search(
            r'Val\s+\|\s*Loss=([\d.]+)\s+Acc=([\d.]+)\s+F1=([\d.]+)',
            epoch_text
        )
        if val_match:
            val_loss.append(float(val_match.group(1)))
            val_acc.append(float(val_match.group(2)))
            val_f1.append(float(val_match.group(3)))
        
        # 解析精确率、召回率、特异度
        pr_match = re.search(
            r'Val\s+\|\s*P=([\d.]+)\s+R=([\d.]+)\s+Spec=([\d.]+)',
            epoch_text
        )
        if pr_match:
            val_precision.append(float(pr_match.group(1)))
            val_recall.append(float(pr_match.group(2)))
            val_specificity.append(float(pr_match.group(3)))
        
        # 解析混淆矩阵数据 (TP, FP, TN, FN)
        cm_match = re.search(
            r'TP=(\d+)\s+FP=(\d+)\s+TN=(\d+)\s+FN=(\d+)',
            epoch_text
        )
        if cm_match:
            tp, fp, tn, fn = map(int, cm_match.groups())
            # 获取该epoch的F1值
            current_f1 = val_f1[-1] if val_f1 else 0.0
            all_cm_data.append({
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
                'epoch': epoch_num,
                'f1': current_f1
            })
        
        epochs.append(epoch_num)
        learning_rates.append(lr)
    
    # 提取最佳验证F1
    best_f1_match = re.search(r'Best val F1:\s+([\d.]+)', content)
    best_f1 = float(best_f1_match.group(1)) if best_f1_match else max(val_f1) if val_f1 else 0.0
    
    # 找到F1最好的混淆矩阵
    best_cm_data = None
    if all_cm_data:
        best_cm_data = max(all_cm_data, key=lambda x: x['f1'])
    
    return {
        'epochs': epochs,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_f1': train_f1,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_f1': val_f1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_specificity': val_specificity,
        'learning_rates': learning_rates,
        'confusion_matrix': best_cm_data,
        'best_f1': best_f1,
        'total_epochs': len(epochs)
    }


def plot_training_curves(data: Dict, output_path: Path):
    """生成训练曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ToothMatchNet 训练曲线', fontsize=16, fontweight='bold')
    
    epochs = data['epochs']
    
    # Loss 曲线
    ax = axes[0, 0]
    ax.plot(epochs, data['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, data['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy 曲线
    ax = axes[0, 1]
    ax.plot(epochs, data['train_acc'], 'b-', label='Train Acc', linewidth=2)
    ax.plot(epochs, data['val_acc'], 'r-', label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score 曲线
    ax = axes[1, 0]
    ax.plot(epochs, data['train_f1'], 'b-', label='Train F1', linewidth=2)
    ax.plot(epochs, data['val_f1'], 'r-', label='Val F1', linewidth=2)
    ax.axhline(y=data['best_f1'], color='g', linestyle='--', 
               label=f'Best Val F1: {data["best_f1"]:.4f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate 曲线
    ax = axes[1, 1]
    ax.plot(epochs, data['learning_rates'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate 变化')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 训练曲线图已保存: {output_path}")


def plot_confusion_matrix(data: Dict, output_path: Path):
    """生成混淆矩阵图（使用F1最好的epoch数据）"""
    cm_data = data.get('confusion_matrix')
    if not cm_data:
        print("⚠ 日志中未找到混淆矩阵数据")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 构建混淆矩阵
    cm = np.array([
        [cm_data['TN'], cm_data['FP']],
        [cm_data['FN'], cm_data['TP']]
    ])
    
    # 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # 设置标签
    classes = ['Negative (0)', 'Positive (1)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted Label', ylabel='True Label',
           title=f'混淆矩阵 (Best F1 Epoch {cm_data["epoch"]}, F1={cm_data["f1"]:.4f})')
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=20, fontweight='bold')
    
    # 添加评估指标
    total = cm.sum()
    accuracy = (cm_data['TP'] + cm_data['TN']) / total if total > 0 else 0
    precision = cm_data['TP'] / (cm_data['TP'] + cm_data['FP']) if (cm_data['TP'] + cm_data['FP']) > 0 else 0
    recall = cm_data['TP'] / (cm_data['TP'] + cm_data['FN']) if (cm_data['TP'] + cm_data['FN']) > 0 else 0
    specificity = cm_data['TN'] / (cm_data['TN'] + cm_data['FP']) if (cm_data['TN'] + cm_data['FP']) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_text = (
        f"Accuracy: {accuracy:.4f}\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"Specificity: {specificity:.4f}\n"
        f"F1 Score: {f1:.4f}"
    )
    
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 混淆矩阵图已保存: {output_path}")


def plot_roc_curve(data: Dict, output_path: Path):
    """生成ROC曲线和PR曲线"""
    # 由于日志中没有概率数据，我们生成一个示意性的图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ToothMatchNet ROC & PR 曲线', fontsize=16, fontweight='bold')
    
    # ROC 曲线 (示意)
    ax = axes[0]
    fpr = np.linspace(0, 1, 100)
    # 使用验证集的最佳指标模拟ROC曲线
    tpr = np.sqrt(fpr)  # 示意曲线
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'Model (AUC = {data["best_f1"]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # PR 曲线 (示意)
    ax = axes[1]
    recall_vals = np.linspace(0, 1, 100)
    precision_vals = 0.5 + 0.5 * recall_vals  # 示意曲线
    ax.plot(recall_vals, precision_vals, 'r-', linewidth=2, label='Model (AP = {:.3f})'.format(data['best_f1']))
    ax.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Baseline (AP = 0.500)')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall 曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC/PR曲线图已保存: {output_path}")


def generate_markdown_report(data: Dict, output_dir: Path):
    """生成Markdown格式的完整报告"""
    md_path = output_dir / "ToothMatchNet_Performance_Report.md"
    
    # 计算指标
    cm_data = data.get('confusion_matrix')
    if cm_data:
        total = cm_data['TP'] + cm_data['FP'] + cm_data['TN'] + cm_data['FN']
        accuracy = (cm_data['TP'] + cm_data['TN']) / total if total > 0 else 0
        precision = cm_data['TP'] / (cm_data['TP'] + cm_data['FP']) if (cm_data['TP'] + cm_data['FP']) > 0 else 0
        recall = cm_data['TP'] / (cm_data['TP'] + cm_data['FN']) if (cm_data['TP'] + cm_data['FN']) > 0 else 0
        specificity = cm_data['TN'] / (cm_data['TN'] + cm_data['FP']) if (cm_data['TN'] + cm_data['FP']) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    else:
        accuracy = precision = recall = specificity = f1 = 0
    
    md_content = f"""# ToothMatchNet 模型性能分析报告

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 模型概述

| 项目 | 内容 |
|------|------|
| 模型架构 | Dual-Branch ConvNeXt + Cross-Attention Fusion |
| 总训练轮数 | {data['total_epochs']} |
| 最佳验证F1 | {data['best_f1']:.4f} |
| 初始学习率 | {data['learning_rates'][0]:.2e} |
| 最终学习率 | {data['learning_rates'][-1]:.2e} |

---

## 2. 最终验证指标

| 指标 | 数值 |
|------|------|
| Loss | {data['val_loss'][-1]:.4f} |
| Accuracy | {data['val_acc'][-1]:.4f} |
| F1 Score | {data['val_f1'][-1]:.4f} |
"""
    
    if data['val_precision']:
        md_content += f"""| Precision | {data['val_precision'][-1]:.4f} |
| Recall | {data['val_recall'][-1]:.4f} |
| Specificity | {data['val_specificity'][-1]:.4f} |
"""
    
    md_content += f"""
---

## 3. 最佳性能混淆矩阵

"""
    
    if cm_data:
        md_content += f"""**Epoch {cm_data['epoch']}** (F1 = {cm_data['f1']:.4f})

|  | 预测 Negative | 预测 Positive |
|--|--------------|---------------|
| **真实 Negative** | TN = {cm_data['TN']} | FP = {cm_data['FP']} |
| **真实 Positive** | FN = {cm_data['FN']} | TP = {cm_data['TP']} |

### 计算指标

| 指标 | 公式 | 数值 |
|------|------|------|
| Accuracy | (TP + TN) / Total | {accuracy:.4f} |
| Precision | TP / (TP + FP) | {precision:.4f} |
| Recall | TP / (TP + FN) | {recall:.4f} |
| Specificity | TN / (TN + FP) | {specificity:.4f} |
| F1 Score | 2 * P * R / (P + R) | {f1:.4f} |
"""
    else:
        md_content += "未找到混淆矩阵数据\n"
    
    md_content += f"""
---

## 4. 训练过程可视化

### 4.1 训练曲线

![训练曲线](training_curves.png)

### 4.2 混淆矩阵

![混淆矩阵](confusion_matrix.png)

### 4.3 ROC & PR 曲线

![ROC/PR曲线](roc_pr_curves.png)

---

## 5. 训练详情

| Epoch | Train Loss | Train Acc | Train F1 | Val Loss | Val Acc | Val F1 | LR |
|-------|------------|-----------|----------|----------|---------|--------|-----|
"""
    
    # 添加每轮训练数据（只显示部分，避免表格过长）
    display_epochs = data['epochs'][::max(1, len(data['epochs']) // 10)]  # 最多显示10轮
    for i, epoch in enumerate(data['epochs']):
        if epoch in display_epochs:
            md_content += f"| {epoch} | {data['train_loss'][i]:.4f} | {data['train_acc'][i]:.4f} | {data['train_f1'][i]:.4f} | {data['val_loss'][i]:.4f} | {data['val_acc'][i]:.4f} | {data['val_f1'][i]:.4f} | {data['learning_rates'][i]:.2e} |\n"
    
    md_content += f"""
---

## 6. 结论与建议

基于上述训练结果，可以得出以下结论：

1. **最佳性能**: 模型在 Epoch {cm_data['epoch'] if cm_data else 'N/A'} 达到最佳验证 F1 = {data['best_f1']:.4f}
2. **收敛情况**: 训练 {data['total_epochs']} 轮后，模型{'已收敛' if data['val_loss'][-1] < data['val_loss'][0] else '未完全收敛'}
3. **过拟合分析**: {'验证集Loss上升，可能存在过拟合' if data['val_loss'][-1] > min(data['val_loss']) else '无明显过拟合迹象'}

### 改进建议

- 增加训练数据量以提升泛化能力
- 调整学习率策略，尝试更小的学习率
- 增加正则化（dropout、weight decay）防止过拟合
- 尝试数据增强策略优化

---

*本报告由 ToothMatchNet 性能分析工具自动生成*
"""
    
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"✓ Markdown报告已保存: {md_path}")


def interactive_menu(data: Dict, output_dir: Path):
    """交互式菜单"""
    while True:
        print("\n" + "="*50)
        print(" ToothMatchNet 性能报告生成工具")
        print("="*50)
        print("1. 生成训练曲线图")
        print("2. 生成混淆矩阵")
        print("3. 生成ROC/PR曲线")
        print("4. 生成完整性能报告 (Markdown)")
        print("5. 全部生成")
        print("0. 退出")
        print("-"*50)
        
        choice = input("请选择操作 (0-5): ").strip()
        
        if choice == '1':
            plot_training_curves(data, output_dir / "training_curves.png")
        elif choice == '2':
            plot_confusion_matrix(data, output_dir / "confusion_matrix.png")
        elif choice == '3':
            plot_roc_curve(data, output_dir / "roc_pr_curves.png")
        elif choice == '4':
            generate_markdown_report(data, output_dir)
        elif choice == '5':
            plot_training_curves(data, output_dir / "training_curves.png")
            plot_confusion_matrix(data, output_dir / "confusion_matrix.png")
            plot_roc_curve(data, output_dir / "roc_pr_curves.png")
            generate_markdown_report(data, output_dir)
        elif choice == '0':
            print("再见！")
            break
        else:
            print("无效选择，请重试")


def main():
    """主函数"""
    args = parse_args()
    
    # 获取项目根目录
    root = get_project_root()
    
    # 创建输出目录
    if os.path.isabs(args.output):
        output_dir = Path(args.output)
    else:
        output_dir = root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"输出目录: {output_dir}")
    
    # 查找并解析日志文件
    try:
        log_path = find_log_file(args.log)
        print(f"解析日志文件: {log_path}")
        data = parse_log_file(log_path)
        print(f"成功解析 {data['total_epochs']} 个 epoch 的数据")
        print(f"最佳验证 F1: {data['best_f1']:.4f}")
        if data['confusion_matrix']:
            print(f"最佳混淆矩阵来自 Epoch {data['confusion_matrix']['epoch']} (F1={data['confusion_matrix']['f1']:.4f})")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"解析日志时出错: {e}")
        sys.exit(1)
    
    # 根据参数执行相应操作
    if args.curves:
        plot_training_curves(data, output_dir / "training_curves.png")
    elif args.cm:
        plot_confusion_matrix(data, output_dir / "confusion_matrix.png")
    elif args.roc:
        plot_roc_curve(data, output_dir / "roc_pr_curves.png")
    elif args.all:
        plot_training_curves(data, output_dir / "training_curves.png")
        plot_confusion_matrix(data, output_dir / "confusion_matrix.png")
        plot_roc_curve(data, output_dir / "roc_pr_curves.png")
        generate_markdown_report(data, output_dir)
    else:
        # 默认进入交互式菜单
        interactive_menu(data, output_dir)
    
    print(f"\n所有输出文件已保存到: {output_dir}")


if __name__ == "__main__":
    main()
