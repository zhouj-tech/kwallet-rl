import numpy as np
import json
import hashlib

# =========================================================
# ✅ 只改这里：生成交易序列的配置（要和你们对齐的 baseline/DQN 参数一致）
# =========================================================
TX_CONFIG = {
    "seed": 123,
    "episodes": 3000,          # 你要生成多少个回合的交易序列
    "steps_per_episode": 1000, # 每回合多少步（通常=你们评估的 max_steps）
    "max_transaction": 1000,   # 交易上限 T / max_tx（要一致）
    # 两种生成方式：
    # 1) "independent": 每回合用不同seed（seed + 100000*ep），回合间独立
    # 2) "continuous": 用同一个rng连续生成所有回合（更像“不断跑下去”的感觉）
    "mode": "independent",
}
# =========================================================

OUT_NPY = "tx_stream.npy"
OUT_META = "tx_meta.json"

# ✅ 新增：给 monte_carlo_fwf_aligned 用的配置（这就是你之前报 KeyError 的根因）
BASELINE_CONFIG = {
    "seed": TX_CONFIG["seed"],
    "env": {
        "C": 3000,
        "k": 3,
        "F": 10,
        "max_transaction": TX_CONFIG["max_transaction"],
        "enable_shaping": True,
    },
    "episode_steps": TX_CONFIG["steps_per_episode"],
    "mc": {
        "runs": 500,                  # 这里是“估期望跑多少回合”，不必=3000
        "independent_episodes": False # 用外部 tx_stream 时，这项影响不大
    }
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_tx_streams(cfg: dict) -> np.ndarray:
    seed = int(cfg["seed"])
    E = int(cfg["episodes"])
    S = int(cfg["steps_per_episode"])
    T = int(cfg["max_transaction"])
    mode = str(cfg["mode"]).lower()

    tx = np.empty((E, S), dtype=np.int32)

    if mode == "independent":
        for ep in range(E):
            rng = np.random.default_rng(seed + 100000 * ep)
            tx[ep, :] = rng.integers(1, T + 1, size=S, dtype=np.int32)
    elif mode == "continuous":
        rng = np.random.default_rng(seed)
        tx[:, :] = rng.integers(1, T + 1, size=(E, S), dtype=np.int32)
    else:
        raise ValueError("mode must be 'independent' or 'continuous'")

    return tx


def main():
    tx = generate_tx_streams(TX_CONFIG)

    np.save(OUT_NPY, tx)

    meta = dict(TX_CONFIG)
    meta["shape"] = [int(tx.shape[0]), int(tx.shape[1])]
    meta["dtype"] = str(tx.dtype)

    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("已生成交易序列文件：")
    print(f"1) {OUT_NPY}  shape={tx.shape}  dtype={tx.dtype}")
    print(f"2) {OUT_META}")
    print("请把这两个文件发给组员。")
    print(f"sha256({OUT_NPY}) = {sha256_file(OUT_NPY)}")
    print("\n预览：第0回合前10个交易 =", tx[0, :10].tolist())


# ===== 与 DQN 完全一致的 shaping 常量 =====
REFRESH_COST = 0.01
IMBALANCE_PENALTY = 0.02
WASTEFUL_REFRESH_PENALTY = 0.02
WASTEFUL_REFRESH_THRESH = 0.6


class KWalletFWFStrictAligned:
    def __init__(self, C, k, F, max_transaction, max_steps, seed, enable_shaping=True):
        self.C = float(C)
        self.k = int(k)
        self.F = int(F)
        self.max_transaction = int(max_transaction)
        self.max_steps = int(max_steps)
        self.enable_shaping = bool(enable_shaping)

        self.wallet_size = self.C / self.k

        self.alpha_drop = 0.02
        self.beta_flush = REFRESH_COST

        self.rng = np.random.default_rng(seed)
        self.reset()

    def _generate_transaction(self):
        return int(self.rng.integers(1, self.max_transaction + 1))

    def _usable(self, i: int) -> bool:
        return self.time > self.freeze_until[i]

    def _next_usable_from(self, start: int):
        for s in range(self.k):
            j = (start + s) % self.k
            if self._usable(j):
                return j
        return None

    def reset(self):
        self.wallets = [self.wallet_size] * self.k
        self.freeze_until = [-1] * self.k
        self.pending_refill = [False] * self.k

        self.total_settled = 0.0
        self.total_accepted = 0.0
        self.num_flushes = 0
        self.drops = 0
        self.oversize_drops = 0

        self.time = 0
        self.idx = 0
        self.G = 0.0

    def step(self, tx: int):
        reward = 0.0
        flushes_this_step = 0
        refresh_targets = []

        pre_refresh_balances = {i: self.wallets[i] for i in range(self.k)}

        chosen_idx = self._next_usable_from(self.idx)

        if tx > self.wallet_size * self.k:
            self.oversize_drops += 1
            self.drops += 1
            reward -= self.alpha_drop
        else:
            if chosen_idx is None:
                self.drops += 1
                reward -= self.alpha_drop
            else:
                if self.wallets[chosen_idx] >= tx:
                    self.wallets[chosen_idx] -= tx
                    self.total_settled += tx
                    self.total_accepted += tx
                    reward += float(tx) / self.max_transaction
                    self.idx = chosen_idx
                else:
                    self.pending_refill[chosen_idx] = True
                    self.wallets[chosen_idx] = 0.0
                    self.freeze_until[chosen_idx] = self.time + self.F - 1
                    self.num_flushes += 1
                    flushes_this_step += 1
                    refresh_targets.append(chosen_idx)

                    self.drops += 1
                    reward -= self.alpha_drop

                    self.idx = (chosen_idx + 1) % self.k

        reward -= self.beta_flush * flushes_this_step

        if self.enable_shaping:
            usable_balances = [self.wallets[i] for i in range(self.k) if self._usable(i)]
            if len(usable_balances) >= 2:
                std_norm = float(np.std(np.array(usable_balances)) / self.wallet_size)
                reward -= IMBALANCE_PENALTY * std_norm

            for i in refresh_targets:
                if (pre_refresh_balances[i] / self.wallet_size) >= WASTEFUL_REFRESH_THRESH:
                    reward -= WASTEFUL_REFRESH_PENALTY

        self.time += 1

        for i in range(self.k):
            if self.pending_refill[i] and self._usable(i):
                self.wallets[i] = self.wallet_size
                self.pending_refill[i] = False

        self.G += float(reward)
        done = (self.time >= self.max_steps)
        return float(reward), bool(done)

    def utilization_proxy(self):
        denom = (self.max_transaction * self.max_steps)
        if denom <= 0:
            return 0.0
        return float(self.total_accepted) / float(denom)


def _mean_std_ci95(x):
    x = np.asarray(x, dtype=float)
    m = float(x.mean()) if x.size else 0.0
    s = float(x.std(ddof=0)) if x.size else 0.0
    ci = 1.96 * s / np.sqrt(max(1, x.size))
    return m, s, ci


# ✅ 只改：加一个 tx_stream 参数；有就用外部序列，没有就用 env 自己随机
def monte_carlo_fwf_aligned(config: dict, tx_stream: np.ndarray | None = None):
    env_cfg = config["env"]
    seed = int(config["seed"])
    max_steps = int(config["episode_steps"])
    mc_runs = int(config["mc"]["runs"])
    independent = bool(config["mc"]["independent_episodes"])

    if tx_stream is not None:
        if tx_stream.ndim != 2:
            raise ValueError("tx_stream 必须是二维数组 (episodes, steps)")
        if tx_stream.shape[1] < max_steps:
            raise ValueError("tx_stream 每回合步数不足 max_steps")
        if mc_runs > tx_stream.shape[0]:
            raise ValueError("mc.runs 不能超过 tx_stream 的回合数")

    returns, settled, accepted, drops, flushes, util, oversize = [], [], [], [], [], [], []

    base_env = None
    if (not independent) and (tx_stream is None):
        base_env = KWalletFWFStrictAligned(
            C=env_cfg["C"],
            k=env_cfg["k"],
            F=env_cfg["F"],
            max_transaction=env_cfg["max_transaction"],
            max_steps=max_steps,
            seed=seed,
            enable_shaping=env_cfg["enable_shaping"],
        )

    for ep in range(mc_runs):
        if tx_stream is not None:
            env = KWalletFWFStrictAligned(
                C=env_cfg["C"],
                k=env_cfg["k"],
                F=env_cfg["F"],
                max_transaction=env_cfg["max_transaction"],
                max_steps=max_steps,
                seed=seed,
                enable_shaping=env_cfg["enable_shaping"],
            )
        else:
            if independent:
                env = KWalletFWFStrictAligned(
                    C=env_cfg["C"],
                    k=env_cfg["k"],
                    F=env_cfg["F"],
                    max_transaction=env_cfg["max_transaction"],
                    max_steps=max_steps,
                    seed=seed + 100000 * ep,
                    enable_shaping=env_cfg["enable_shaping"],
                )
            else:
                env = base_env
                env.reset()

        for t in range(max_steps):
            tx = int(tx_stream[ep, t]) if tx_stream is not None else env._generate_transaction()
            _, done = env.step(tx)
            if done:
                break

        returns.append(env.G)
        settled.append(env.total_settled)
        accepted.append(env.total_accepted)
        drops.append(env.drops)
        flushes.append(env.num_flushes)
        util.append(env.utilization_proxy())
        oversize.append(env.oversize_drops)

    mG, sG, ciG = _mean_std_ci95(returns)
    mS, sS, ciS = _mean_std_ci95(settled)
    mA, sA, ciA = _mean_std_ci95(accepted)
    mD, sD, ciD = _mean_std_ci95(drops)
    mF, sF, ciF = _mean_std_ci95(flushes)
    mU, sU, ciU = _mean_std_ci95(util)
    mO, sO, ciO = _mean_std_ci95(oversize)

    drop_rates = np.asarray(drops, dtype=float) / float(max_steps)
    mDR, sDR, ciDR = _mean_std_ci95(drop_rates)

    print("\n================= Monte Carlo 估期望（Strict FWF，对齐 DQN 口径）=================")
    print(f"• 配置：C={env_cfg['C']}, k={env_cfg['k']}, F={env_cfg['F']}, max_tx={env_cfg['max_transaction']}, steps/回合={max_steps}, 回合数N={mc_runs}")
    print(f"• Reward口径：完全照 DQN env.step()（接收=tx/max_tx；drop=-0.02；flush成本=-0.01*次数；shaping={env_cfg['enable_shaping']}）")
    print("-------------------------------------------------------------------")
    print(f"• Return(G) 期望：{mG:.3f} ± {sG:.3f}（95%CI ± {ciG:.3f}）")
    print(f"• Total Settled 期望：{mS:.1f} ± {sS:.1f}（95%CI ± {ciS:.1f}）")
    print(f"• Total Accepted 期望：{mA:.1f} ± {sA:.1f}（95%CI ± {ciA:.1f}）")
    print(f"• Drops 期望：{mD:.1f} ± {sD:.1f}（95%CI ± {ciD:.1f}）")
    print(f"• DropRate 期望：{mDR*100:.2f}% ± {sDR*100:.2f}%（95%CI ± {ciDR*100:.2f}%）")
    print(f"• Flushes 期望：{mF:.1f} ± {sF:.1f}（95%CI ± {ciF:.1f}）")
    print(f"• UtilizationProxy 期望：{mU*100:.2f}% ± {sU*100:.2f}%（95%CI ± {ciU*100:.2f}%）")
    print(f"• OversizeDrops 期望：{mO:.2f} ± {sO:.2f}（95%CI ± {ciO:.2f}）")
    print("===================================================================\n")

    return {
        "returns": np.asarray(returns, dtype=float),
        "settled": np.asarray(settled, dtype=float),
        "accepted": np.asarray(accepted, dtype=float),
        "drops": np.asarray(drops, dtype=float),
        "flushes": np.asarray(flushes, dtype=float),
        "util": np.asarray(util, dtype=float),
        "oversize": np.asarray(oversize, dtype=float),
    }


if __name__ == "__main__":
    # 1) 先生成并保存交易序列（给组员用）
    main()

    # 2) 再用同一条交易序列跑 FWF baseline（真正做到“同序列对齐”）
    tx_stream = np.load(OUT_NPY)
    monte_carlo_fwf_aligned(BASELINE_CONFIG, tx_stream=tx_stream)
