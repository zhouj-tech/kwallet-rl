import os
import datetime 
import numpy as np
import json
import hashlib

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional

from pathlib import Path
def _set_chinese_font():
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",          # macOS
        "/System/Library/Fonts/STHeiti Medium.ttc",    # macOS older
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Linux
        "C:/Windows/Fonts/simhei.ttf",                 # Windows
    ]
    for p in candidates:
        if Path(p).exists():
            font_prop = fm.FontProperties(fname=p)
            plt.rcParams["font.family"] = font_prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            print(f"✅ 使用中文字体: {p}")
            return
    raise RuntimeError("❌ 未找到可用的中文字体，无法绘制中文")


# ========================================================= 
# ✅ 统一参数设置 (与 DQN 完全一致)
# =========================================================
CONFIG = {
    "seed": 123,
    "env": {
        "C": 3000.0,
        "k": 3,
        "T": 1000,
        "F": 1,
    },
    "eval": {
        "num_episodes": 100,
        "max_steps": 1000,
        "tx_pool_path": "shared_tx_pool.npy",
        "test_start_idx": 4000  # 从第4000个episode开始测试
    },
    "output": {
        "save_results": True,
        "results_path": "baseline_results.json",
        "plot_path": "baseline_evaluation.png"
    }
}


class KWalletFWFStrict:
    """
    严格的 First-Write-First (FWF) 策略实现
    
    策略描述:
    - 按照钱包索引顺序尝试结算交易
    - 如果当前钱包余额不足,刷新该钱包并丢弃交易
    - 交易金额超过单个钱包容量时直接丢弃
    """
    
    def __init__(self, C: float, k: int, T: int, F: int, max_steps: int, seed: int = 123):
        """
        参数:
            C: 总资金量
            k: 钱包数量
            T: 最大交易金额
            F: 刷新冻结期
            max_steps: 最大步数
            seed: 随机种子(虽然FWF是确定性策略,但保留用于对齐)
        """
        self.C = float(C)
        self.k = int(k)
        self.size = self.C / self.k  # 单个钱包容量
        self.T = int(T)
        self.F = int(F)
        self.max_steps = int(max_steps)
        self.seed = seed
        self.reset_state()
    
    def reset_state(self):
        """重置所有内部状态"""
        self.t = 0
        self.wallets = np.full(self.k, self.size, dtype=float)
        self.freeze_until = np.full(self.k, -1, dtype=int)
        self.idx = 0  # 当前尝试的钱包索引
        
        # 业务指标
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.num_drops = 0
        self.oversize_drops = 0  # 超大交易丢包
        self.insufficient_drops = 0  # 余额不足丢包
        
        # 详细记录(用于调试和分析)
        self.transaction_history = []
    
    def _usable(self, i: int) -> bool:
        """检查钱包i是否可用(未冻结)"""
        return self.t > self.freeze_until[i]
    
    def _next_usable_from(self, start: int) -> Optional[int]:
        """从start开始查找下一个可用钱包"""
        for s in range(self.k):
            j = (start + s) % self.k
            if self._usable(j):
                return j
        return None
    
    def reset(self, tx_stream: list[int]):
        """
        重置环境并加载新的交易流
        
        参数:
            tx_stream: 交易序列(长度应为max_steps)
        """
        self.reset_state()
        self.tx_stream = list(tx_stream[:self.max_steps])
        
        if len(self.tx_stream) < self.max_steps:
            raise ValueError(
                f"交易流长度不足: 需要 {self.max_steps}, 实际 {len(self.tx_stream)}"
            )
        
        self.current_tx = self.tx_stream[self.t]
    
    def step(self) -> float:
        """
        执行一步FWF策略
        
        返回:
            accepted: 本步接受的交易金额(0表示丢包)
        """
        tx = self.current_tx
        accepted = 0.0
        action_taken = None
        
        # 检查交易是否超过单个钱包容量
        if tx > self.size:
            self.num_drops += 1
            self.oversize_drops += 1
            action_taken = "DROP_OVERSIZE"
        else:
            # 寻找下一个可用钱包
            i = self._next_usable_from(self.idx)
            
            if i is None:
                # 所有钱包都被冻结
                self.num_drops += 1
                self.insufficient_drops += 1
                action_taken = "DROP_ALL_FROZEN"
            else:
                # 尝试在钱包i中结算
                if self.wallets[i] >= tx:
                    # 余额充足,成功结算
                    self.wallets[i] -= tx
                    accepted = float(tx)
                    self.total_accepted += accepted
                    self.idx = i  # 更新索引
                    action_taken = f"SETTLE_W{i}"
                else:
                    # 余额不足,刷新钱包并丢包
                    self.wallets[i] = self.size
                    self.freeze_until[i] = self.t + self.F
                    self.num_flushes += 1
                    self.num_drops += 1
                    self.insufficient_drops += 1
                    self.idx = (i + 1) % self.k  # 移动到下一个钱包
                    action_taken = f"FLUSH_W{i}_DROP"
        
        # 记录详细信息(可选,用于调试)
        self.transaction_history.append({
            'step': self.t,
            'tx': tx,
            'accepted': accepted,
            'action': action_taken,
            'wallets': self.wallets.copy(),
        })
        
        # 推进时间
        self.t += 1
        
        # 更新冻结状态
        for i in range(self.k):
            if self.freeze_until[i] == self.t:
                self.wallets[i] = self.size
        
        # 加载下一笔交易
        if self.t < self.max_steps:
            self.current_tx = self.tx_stream[self.t]
        
        return accepted
    
    def get_metrics(self) -> dict[str, float]:
        """
        获取与DQN一致的业务指标
        
        返回:
            metrics: 包含所有关键业务指标的字典
        """
        return {
            'settled': self.total_accepted,  # 总处理金额
            'drops': self.num_drops,  # 总丢包数
            'oversize_drops': self.oversize_drops,  # 超大交易丢包
            'insufficient_drops': self.insufficient_drops,  # 余额不足丢包
            'flushes': self.num_flushes,  # 刷新次数
            'utilization': self.total_accepted / (self.C * self.max_steps),  # 资金利用率
            'avg_tx_value': self.total_accepted / max(1, self.max_steps - self.num_drops),  # 平均处理交易额
            'drop_rate': self.num_drops / self.max_steps,  # 丢包率
        }


def verify_data_integrity(tx_pool_path: str, config: dict) -> bool:
    """
    验证数据文件的完整性和正确性
    
    返回:
        is_valid: 数据是否有效
    """
    print("\n" + "="*70)
    print("🔍 数据完整性验证")
    print("="*70)
    
    if not os.path.exists(tx_pool_path):
        print(f"❌ 错误: 找不到文件 {tx_pool_path}")
        print("   请先运行 generate_shared_data.py 生成交易流")
        return False
    
    try:
        # 加载数据
        tx_pool = np.load(tx_pool_path)
        file_size = os.path.getsize(tx_pool_path) / 1024  # KB
        
        print(f"✅ 成功加载文件: {tx_pool_path}")
        print(f"📊 矩阵形状 (Episodes, Steps): {tx_pool.shape}")
        print(f"💾 文件大小: {file_size:.2f} KB")
        
        # 验证尺寸
        expected_steps = config["eval"]["max_steps"]
        if tx_pool.shape[1] != expected_steps:
            print(f"⚠️  警告: 步数不匹配 (期望 {expected_steps}, 实际 {tx_pool.shape[1]})")
        
        # 验证数值范围
        max_val = np.max(tx_pool)
        min_val = np.min(tx_pool)
        print(f"📈 交易金额范围: [{min_val}, {max_val}]")
        
        if max_val > config["env"]["T"]:
            print(f"⚠️  警告: 存在超过T={config['env']['T']}的交易")
        
        # 计算测试段指纹
        test_start = config["eval"]["test_start_idx"]
        test_end = test_start + config["eval"]["num_episodes"]
        
        if test_end > tx_pool.shape[0]:
            print(f"❌ 错误: 测试范围超出数据集 ({test_end} > {tx_pool.shape[0]})")
            return False
        
        test_segment = tx_pool[test_start:test_end]
        segment_hash = hashlib.md5(test_segment.tobytes()).hexdigest()
        
        print(f"\n🔑 测试段数据指纹 (MD5): {segment_hash}")
        print(f"📍 测试段范围: Episodes [{test_start}, {test_end})")
        print(f"🎲 测试段首行前5笔交易: {test_segment[0, :5].tolist()}")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {str(e)}")
        return False


def run_evaluation(config: dict) -> dict[str, any]:
    """
    执行完整的评估流程
    
    返回:
        results: 包含所有评估结果的字典
    """
    # 验证数据
    tx_pool_path = config["eval"]["tx_pool_path"]
    if not verify_data_integrity(tx_pool_path, config):
        raise RuntimeError("数据验证失败,终止评估")
    
    # 加载测试数据
    tx_pool = np.load(tx_pool_path)
    test_start = config["eval"]["test_start_idx"]
    test_end = test_start + config["eval"]["num_episodes"]
    test_segment = tx_pool[test_start:test_end]
    
    # 初始化环境
    env = KWalletFWFStrict(
        C=config["env"]["C"],
        k=config["env"]["k"],
        T=config["env"]["T"],
        F=config["env"]["F"],
        max_steps=config["eval"]["max_steps"],
        seed=config["seed"]
    )
    
    # 执行评估
    print("🚀 开始评估...")
    print(f"   策略: First-Write-First (FWF)")
    print(f"   回合数: {config['eval']['num_episodes']}")
    print(f"   每回合步数: {config['eval']['max_steps']}\n")
    
    all_results = []
    
    for ep in range(config["eval"]["num_episodes"]):
        # 重置环境并加载当前回合的交易流
        env.reset(tx_stream=test_segment[ep])
        
        # 执行完整回合
        for step in range(config["eval"]["max_steps"]):
            env.step()
        
        # 收集指标
        metrics = env.get_metrics()
        all_results.append(metrics)
        
        # 进度显示
        if (ep + 1) % 20 == 0:
            print(f"   进度: {ep + 1}/{config['eval']['num_episodes']} 回合完成")
    
    print("✅ 评估完成!\n")
    
    # 汇总统计
    summary = {}
    metric_names = all_results[0].keys()
    
    for metric in metric_names:
        values = [r[metric] for r in all_results]
        summary[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'values': values  # 保留原始数据用于后续分析
        }
    
    return {
        'config': config,
        'timestamp': datetime.datetime.now().isoformat(),
        'num_episodes': config["eval"]["num_episodes"],
        'summary': summary,
        'raw_results': all_results
    }


def print_evaluation_report(results: dict[str, any]):
    """打印格式化的评估报告"""
    summary = results['summary']
    
    print("\n" + "="*70)
    print("📊 FWF Baseline 评估报告")
    print("="*70)
    print(f"⏰ 评估时间: {results['timestamp']}")
    print(f"🎯 测试回合数: {results['num_episodes']}")
    print("-"*70)
    
    # 主要业务指标
    print("\n【核心业务指标】")
    print(f"{'指标':<30} {'均值':>12} {'标准差':>12} {'范围':>20}")
    print("-"*70)
    
    key_metrics = ['settled', 'drops', 'flushes', 'utilization', 'drop_rate']
    
    for metric in key_metrics:
        if metric in summary:
            data = summary[metric]
            mean = data['mean']
            std = data['std']
            min_val = data['min']
            max_val = data['max']
            
            # 特殊格式化
            if metric == 'utilization':
                print(f"{'资金利用率 (%)':<30} {mean*100:>12.2f} {std*100:>12.2f} [{min_val*100:.2f}, {max_val*100:.2f}]")
            elif metric == 'drop_rate':
                print(f"{'丢包率 (%)':<30} {mean*100:>12.2f} {std*100:>12.2f} [{min_val*100:.2f}, {max_val*100:.2f}]")
            else:
                label_map = {
                    'settled': '总处理金额',
                    'drops': '丢包数',
                    'flushes': '刷新次数'
                }
                label = label_map.get(metric, metric)
                print(f"{label:<30} {mean:>12.2f} {std:>12.2f} [{min_val:.2f}, {max_val:.2f}]")
    
    # 扩展指标
    print("\n【扩展分析指标】")
    print(f"{'指标':<30} {'值':>12}")
    print("-"*70)
    
    extended_metrics = ['avg_tx_value', 'oversize_drops', 'insufficient_drops']
    for metric in extended_metrics:
        if metric in summary:
            mean = summary[metric]['mean']
            label_map = {
                'avg_tx_value': '平均处理交易额',
                'oversize_drops': '超大交易丢包数',
                'insufficient_drops': '余额不足丢包数'
            }
            label = label_map.get(metric, metric)
            print(f"{label:<30} {mean:>12.2f}")
    
    print("="*70 + "\n")


def plot_evaluation_results(results: dict[str, any], save_path: str):
    """生成可视化图表"""
    summary = results['summary']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FWF Baseline 评估结果分析', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('settled', '总处理金额', 'C0'),
        ('drops', '丢包数', 'C1'),
        ('flushes', '刷新次数', 'C2'),
        ('utilization', '资金利用率', 'C3'),
        ('drop_rate', '丢包率', 'C4'),
        ('avg_tx_value', '平均处理交易额', 'C5')
    ]
    
    for idx, (metric, label, color) in enumerate(metrics_to_plot):
        if metric not in summary:
            continue
            
        ax = axes[idx // 3, idx % 3]
        values = summary[metric]['values']
        
        # 绘制分布
        ax.hist(values, bins=30, alpha=0.7, color=color, edgecolor='black')
        
        # 添加统计线
        mean = summary[metric]['mean']
        median = summary[metric]['median']
        ax.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(median, color='green', linestyle=':', linewidth=2, label=f'Median: {median:.2f}')
        
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('频数', fontsize=11)
        ax.set_title(f'{label} 分布', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 可视化图表已保存至: {save_path}")


def save_results(results: dict[str, any], save_path: str):
    """保存结果为JSON格式(移除原始数据以减小文件大小)"""
    # 创建精简版本(移除原始values数组)
    compact_results = {
        'config': results['config'],
        'timestamp': results['timestamp'],
        'num_episodes': results['num_episodes'],
        'summary': {}
    }
    
    for metric, data in results['summary'].items():
        compact_results['summary'][metric] = {
            'mean': data['mean'],
            'std': data['std'],
            'min': data['min'],
            'max': data['max'],
            'median': data['median']
        }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 评估结果已保存至: {save_path}")


def compare_with_dqn(baseline_results: dict[str, any], dqn_results_path: str = "dqn_results.json"):
    """
    与DQN结果进行对比分析(如果DQN结果存在)
    
    参数:
        baseline_results: Baseline评估结果
        dqn_results_path: DQN结果文件路径
    """
    if not os.path.exists(dqn_results_path):
        print(f"\n💡 提示: 未找到DQN结果文件 ({dqn_results_path})")
        print("   若要进行对比分析,请先运行DQN评估并保存结果\n")
        return
    
    try:
        with open(dqn_results_path, 'r') as f:
            dqn_results = json.load(f)
        
        print("\n" + "="*70)
        print("⚔️  Baseline vs DQN 对比分析")
        print("="*70)
        
        baseline_summary = baseline_results['summary']
        dqn_summary = dqn_results['summary']
        
        print(f"{'指标':<25} {'Baseline':>15} {'DQN':>15} {'提升率':>15}")
        print("-"*70)
        
        compare_metrics = ['settled', 'drops', 'flushes', 'utilization']
        
        for metric in compare_metrics:
            if metric in baseline_summary and metric in dqn_summary:
                baseline_val = baseline_summary[metric]['mean']
                dqn_val = dqn_summary[metric]['mean']
                
                # 计算提升率(注意drops和flushes越低越好)
                if metric in ['drops', 'flushes']:
                    improvement = (baseline_val - dqn_val) / baseline_val * 100
                else:
                    improvement = (dqn_val - baseline_val) / baseline_val * 100
                
                # 格式化输出
                if metric == 'utilization':
                    print(f"{'资金利用率':<25} {baseline_val*100:>14.2f}% {dqn_val*100:>14.2f}% {improvement:>14.2f}%")
                else:
                    label_map = {
                        'settled': '总处理金额',
                        'drops': '丢包数',
                        'flushes': '刷新次数'
                    }
                    label = label_map.get(metric, metric)
                    print(f"{label:<25} {baseline_val:>15.2f} {dqn_val:>15.2f} {improvement:>14.2f}%")
                
                # 统计显著性检验
                baseline_values = baseline_summary[metric]['values']
                dqn_values = dqn_summary[metric]['values']
                t_stat, p_value = stats.ttest_ind(baseline_values, dqn_values)
                
                sig_mark = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
                print(f"  └─ 统计检验: t={t_stat:.3f}, p={p_value:.4f} {sig_mark}")
        
        print("="*70)
        print("注: *** p<0.001, ** p<0.01, * p<0.05")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"⚠️  对比分析失败: {str(e)}\n")


# =========================================================
# 主执行流程
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🎯 K-Wallet FWF Baseline 评估系统")
    print("="*70)
    
    try:
        # 1. 执行评估
        results = run_evaluation(CONFIG)
        
        # 2. 打印报告
        print_evaluation_report(results)
        
        # 3. 保存结果
        if CONFIG["output"]["save_results"]:
            save_results(results, CONFIG["output"]["results_path"])
        
        # 4. 生成可视化
        _set_chinese_font()

        plot_evaluation_results(results, CONFIG["output"]["plot_path"])
        
        # 5. 与DQN对比(如果存在)
        compare_with_dqn(results)
        
        print("\n✅ 所有评估任务完成!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 评估过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
