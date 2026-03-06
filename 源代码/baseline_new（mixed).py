import numpy as np
from typing import List, Optional, Literal

class KWalletFWF:
  

    def __init__(self,
                 C: float = 10000.0,
                 k: int = 10,
                 T: int = 100,
                 F: int = 10,
                 max_steps: int = 2000,
                 seed: int = 123,
                 normalize_reward: bool = True,
                 verbose: bool = False,
                 snapshot_every: int = 0,
                 style: Literal['plain','research'] = 'research'):
        self.C = float(C)
        self.k = int(k)
        self.size = self.C / self.k
        self.T = int(T)
        self.F = int(F)
        self.max_steps = int(max_steps)
        self.normalize_reward = bool(normalize_reward)
        self.verbose = verbose
        self.snapshot_every = int(snapshot_every)
        self.style = style
        self.rng = np.random.default_rng(seed)

        # 运行期状态
        self.t = 0
        self.wallets = None
        self.freeze_until = None
        self.idx = 0
        self.active_idx = 0
        self.wait_until = -1
        self.current_tx = None

        # 统计
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.num_drops = 0
        self.num_oversize_drops = 0
        self.wait_steps = 0
        self.history = {
            'accepted': [], 'tx': [], 'placed': [], 'flushes_step': [], 'drop': [], 'log': []
        }

    # --------------------------------------------------
    def _gen_tx(self) -> int:
        return int(self.rng.integers(1, self.T + 1))

    def _log(self, msg: str):
        if self.style == 'research':
            line = f"[t={self.t:03d}] {msg}"
        else:
            line = f"时间 {self.t:>4d}: {msg}"
        self.history['log'].append(line)
        if self.verbose:
            print(line)

    def _utilization_now(self) -> float:
        used = self.C - float(np.sum(self.wallets))
        return max(0.0, min(1.0, used / self.C))

    def _snapshot(self):
        if not self.snapshot_every:
            return
        if self.t % self.snapshot_every != 0 or self.t == 0:
            return
        active = int(np.sum(self.t >= self.freeze_until))
        avg_bal = float(np.mean(self.wallets))
        self._log(f"—— 状态快照 —— 活跃:{active}/{self.k} | 平均余额:{avg_bal:.1f} | Flush累计:{self.num_flushes} | 当前利用率:{self._utilization_now():.3f}")

    def reset(self, tx_stream: Optional[List[int]] = None,reset_seed: bool = False):
        if reset_seed:
           self.rng = np.random.default_rng()

        self.t = 0
        self.wallets = np.full(self.k, self.size, dtype=float)
        self.freeze_until = np.full(self.k, -1, dtype=int)
        self.idx = 0
        self.active_idx = 0
        self.wait_until = -1
        self.num_flushes = 0
        self.num_drops = 0
        self.num_oversize_drops = 0
        self.total_accepted = 0.0
        self.wait_steps = 0
        for k in self.history:
            self.history[k].clear()
        if tx_stream is None:
            self.tx_stream = [self._gen_tx() for _ in range(self.max_steps)]
        else:
            if len(tx_stream) < self.max_steps:
                raise ValueError("tx_stream length < max_steps")
            self.tx_stream = list(tx_stream[:self.max_steps])
        self.current_tx = self.tx_stream[self.t]

    def _flush_wallet(self, i: int):
        self.wallets[i] = self.size
        self.freeze_until[i] = self.t + self.F
        self.num_flushes += 1
        self._log(f"⚠️ 对钱包 {i} 执行 FLUSH → 冻结至 t={self.freeze_until[i]}")

    def _usable(self, i: int) -> bool:
        return self.t >= self.freeze_until[i]

    # --------------------------------------------------
    def step(self):
        tx = self.tx_stream[self.t]
        flushes_this_step = 0
        accepted = 0.0
        placed = False
        self._log(f"🧾 交易到达 {tx}")

        if tx > self.size:
            self.num_oversize_drops += 1
            self.num_drops += 1
            self._log(f"❌ 单笔超过容量（>{self.size:.0f}），丢弃")
        else:
            for step_try in range(self.k):
                    i = (self.idx + step_try) % self.k
                    if not self._usable(i):
                        self._log(f"⏩ 钱包 {i} 冻结（至 t={self.freeze_until[i]}），跳过")
                        continue
                    if self.wallets[i] < tx:
                        self._log(f"⚠️ 钱包 {i} 余额不足（{self.wallets[i]:.0f} < {tx}），触发 FLUSH")
                        self._flush_wallet(i)
                        flushes_this_step += 1
                        continue
                    self.wallets[i] -= tx
                    accepted = float(tx)
                    placed = True
                    self.total_accepted += accepted
                    self._log(f"✅ 钱包 {i} 接单 {tx:>3d} | 余额→ {self.wallets[i]:.0f} | 活跃:{int(np.sum(self.t >= self.freeze_until))}/{self.k}")
                    self.idx = (i + 1) % self.k
                    break
            if not placed:
                    self.num_drops += 1
                    self._log("❌ 遍历一圈仍不可用，丢弃该笔")
                    self._log("ℹ️ 已 flush 并切换活动钱包，本笔丢弃（下一步处理新交易）")

        self._snapshot()

        self.t += 1
        if self.t < self.max_steps:
            self.current_tx = self.tx_stream[self.t]
        self.history['accepted'].append(accepted)
        self.history['tx'].append(tx)
        self.history['placed'].append(int(placed))
        self.history['flushes_step'].append(flushes_this_step)
        self.history['drop'].append(int(not placed))
        return accepted, placed, flushes_this_step

    # ------------------------- 运行-------------------------
    def run_episode(self, tx_stream: Optional[List[int]] = None,reset_seed: bool = False):
        self.reset(tx_stream,reset_seed=reset_seed)

        for _ in range(self.max_steps):
            self.step()
        total_tx = len(self.history['tx'])
        drops = int(np.sum(self.history['drop']))
        util = self.total_accepted / (self.T * total_tx) if self.normalize_reward else self.total_accepted / (self.size * self.k)
        r = self.k * self.T / self.C
        stats = {
            'total_accepted': self.total_accepted,
            'total_steps': total_tx,
            'drops': drops,
            'oversize_drops': self.num_oversize_drops,
            'drop_rate': drops / max(1, total_tx),
            'num_flushes': self.num_flushes,
            'avg_tx': float(np.mean(self.history['tx'])) if total_tx > 0 else 0.0,
            'avg_accepted': float(np.mean(self.history['accepted'])) if total_tx > 0 else 0.0,
            'utilization_proxy': util,
            'offline_ratio': self.wait_steps / max(1, total_tx),
            'r': r,
            'alpha_FWF_theory': (self.k + 1) / (self.k * (1 - r)) if r < 1 else float('inf')
        }

        if self.verbose:
            print("\n=== Summary（研究日志）===")
            active = int(np.sum(self.t >= self.freeze_until))
            avg_bal = float(np.mean(self.wallets))
            print(f"步数: {total_tx} | Flush: {self.num_flushes} | 丢单: {drops} | 活跃:{active}/{self.k}")
            print(f"平均余额: {avg_bal:.1f} | 当前利用率: {self._utilization_now():.3f} | 吞吐proxy: {util:.3f}")
            print(f"r=kT/C: {r:.3f} | α_FWF(理论): {stats['alpha_FWF_theory']:.3f}")

        return stats
    def watch_run(self, steps: int = 50, delay: float = 0.1):
     import time, os
     self.reset(reset_seed=True)
     for _ in range(min(steps, self.max_steps)):
        os.system('clear')  # Windows用'cls'
        print(f"Step {self.t}")
        print("钱包余额: ", [f"{b:.0f}" for b in self.wallets])
        print("冻结状态: ", [int(self.t < f) for f in self.freeze_until])
        print("-" * 40)
        self.step()
        time.sleep(delay)



if __name__ == '__main__':
    env = KWalletFWF(C=10000, k=10, T=300, F=10, max_steps=300, seed=123,
                    verbose=True, snapshot_every=20, style='research')
    #env.watch_run(steps=100, delay=0.2)

    stats = env.run_episode(reset_seed=True)

    print("\n=== 指标 ===")
    for k, v in stats.items():
        print(f"{k:18s}: {v}")
