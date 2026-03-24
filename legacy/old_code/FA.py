import numpy as np
from typing import List, Optional, Literal

class KWalletFA_Paper:
    def __init__(self,
                 C: float = 10000.0,
                 k: int = 10,
                 T: int = 1000,
                 F: int = 3,
                 max_steps: int = 2000,
                 seed: int = 126,
                 normalize_reward: bool = True,
                 verbose: bool = False,
                 snapshot_every: int = 0,
                 style: Literal['plain','research'] = 'research'):

        self.C = float(C)
        self.k = int(k)
        self.size = self.C / self.k          # per-wallet capacity
        self.T = int(T)                      # tx size upper bound
        self.F = int(F)                      # cooldown length
        self.max_steps = int(max_steps)
        self.normalize_reward = bool(normalize_reward)
        self.verbose = bool(verbose)
        self.snapshot_every = int(snapshot_every)
        self.style = style
        self.rng = np.random.default_rng(seed)

        # runtime state
        self.t = 0
        self.wallets = None
        self.freeze_until = None
        self.current_tx = None

        # stats
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
        line = f"[t={self.t:03d}] {msg}" if self.style == 'research' else f"时间 {self.t:>4d}: {msg}"
        self.history['log'].append(line)
        if self.verbose:
            print(line)

    def _usable(self, i: int) -> bool:
        return self.t >= self.freeze_until[i]

    # --------------------------------------------------
    def reset(self, tx_stream: Optional[List[int]] = None, reset_seed: bool = False):
        if reset_seed:
            self.rng = np.random.default_rng()

        self.t = 0
        self.wallets = np.full(self.k, self.size, dtype=float)
        self.freeze_until = np.full(self.k, -1, dtype=int)

        self.total_accepted = 0.0
        self.num_flushes = 0
        self.num_drops = 0
        self.num_oversize_drops = 0
        self.wait_steps = 0

        for key in self.history:
            self.history[key].clear()

        if tx_stream is None:
            self.tx_stream = [self._gen_tx() for _ in range(self.max_steps)]
        else:
            if len(tx_stream) < self.max_steps:
                raise ValueError("tx_stream length < max_steps")
            self.tx_stream = list(tx_stream[:self.max_steps])

        self.current_tx = self.tx_stream[self.t]

    # --------------------------------------------------
    def _flush_all(self):
        for i in range(self.k):
            self.wallets[i] = self.size
            self.freeze_until[i] = self.t + self.F
        self.num_flushes += 1
        self._log(f"⚠️ FLUSH ALL → all wallets frozen until t={self.t + self.F}")

    # --------------------------------------------------
    def step(self):
        tx = self.tx_stream[self.t]
        accepted = 0.0
        placed = False
        flushes_this_step = 0

        self._log(f"🧾 tx arrives: {tx}")

        # Oversize tx: cannot fit any wallet ever
        if tx > self.size:
            self.num_oversize_drops += 1
            self.num_drops += 1
            self._log("❌ oversize tx → DROP")
        else:
            # If ANY wallet is cooling, FA still checks usability wallet-by-wallet
            # But if ALL wallets are cooling, system is offline
            usable_wallets = [i for i in range(self.k) if self._usable(i)]
            if not usable_wallets:
                self.wait_steps += 1
                self.num_drops += 1
                self._log("⏳ all wallets cooling → system offline → DROP")
            else:
                # First-fit across all usable wallets
                for i in usable_wallets:
                    if self.wallets[i] >= tx:
                        self.wallets[i] -= tx
                        accepted = float(tx)
                        placed = True
                        self.total_accepted += accepted
                        self._log(f"✅ insert tx {tx} into wallet {i} | balance→{self.wallets[i]:.0f}")
                        break

                # If no wallet can fit → FLUSH ALL
                if not placed:
                    self._log("⚠️ no wallet can fit → FLUSH ALL")
                    self._flush_all()
                    flushes_this_step += 1

                    # After flush, system enters cooldown → this tx is dropped
                    self.num_drops += 1
                    self._log("❌ tx dropped due to global flush")

        # bookkeeping
        self.t += 1
        if self.t < self.max_steps:
            self.current_tx = self.tx_stream[self.t]

        self.history['accepted'].append(accepted)
        self.history['tx'].append(tx)
        self.history['placed'].append(int(placed))
        self.history['flushes_step'].append(flushes_this_step)
        self.history['drop'].append(int(not placed))

        return accepted, placed, flushes_this_step

    # --------------------------------------------------
    def run_episode(self, tx_stream: Optional[List[int]] = None, reset_seed: bool = False):
        self.reset(tx_stream=tx_stream, reset_seed=reset_seed)

        for _ in range(self.max_steps):
            self.step()

        total_tx = len(self.history['tx'])
        drops = int(np.sum(self.history['drop']))
        util = self.total_accepted / (self.T * total_tx) if self.normalize_reward else self.total_accepted / self.C
        r = self.k * self.T / self.C

        stats = {
            'total_accepted': self.total_accepted,
            'total_steps': total_tx,
            'drops': drops,
            'oversize_drops': self.num_oversize_drops,
            'drop_rate': drops / max(1, total_tx),
            'num_flushes': self.num_flushes,
            'avg_tx': float(np.mean(self.history['tx'])) if total_tx else 0.0,
            'avg_accepted': float(np.mean(self.history['accepted'])) if total_tx else 0.0,
            'utilization_proxy': util,
            'offline_ratio': self.wait_steps / max(1, total_tx),
            'r': r
        }

        if self.verbose:
            print("\n=== Summary (FlushAll paper baseline) ===")
            for k, v in stats.items():
                print(f"{k:18s}: {v}")

        return stats


if __name__ == "__main__":
    env = KWalletFA_Paper(C=10000, k=10, T=1000, F=3,
                          max_steps=2000, seed=123, verbose=False)
    stats = env.run_episode()
    print(stats)
