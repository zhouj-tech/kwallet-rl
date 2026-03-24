import numpy as np
import os
from pathlib import Path

T = 1000
EPISODES = 5000
STEPS_PER_EP = 1000
SEED = 123

# 假设脚本在 kwallet-rl/源代码/ 下
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_tx_pool():
    print("=" * 60)
    print("🚀 开始生成均匀分布交易序列")
    print("=" * 60)

    print(f"T (最大交易金额): {T}")
    print(f"Episodes: {EPISODES}")
    print(f"Steps per episode: {STEPS_PER_EP}")
    print(f"Seed: {SEED}")

    pool_dir = PROJECT_ROOT / "data" / "pools"
    report_dir = PROJECT_ROOT / "data" / "reports" / f"tx_pool_uniform_T{T}"

    os.makedirs(pool_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    filename = f"tx_pool_uniform_T{T}.npy"
    save_path = pool_dir / filename

    rng = np.random.default_rng(SEED)

    all_tx = rng.integers(
        1,
        T + 1,
        size=(EPISODES, STEPS_PER_EP)
    )

    np.save(save_path, all_tx)

    print("\n✅ 生成完成")
    print(f"📁 文件路径: {save_path}")
    print(f"📊 矩阵形状: {all_tx.shape}")

    print("\n" + "=" * 50)
    print("📊 交易流对齐指纹")
    print("=" * 50)
    print("1️⃣ 最初 5 个交易 (EP 0, Step 0-4):")
    print(all_tx[0][:5].tolist())

    print("-" * 30)
    print("2️⃣ 中间交易 (EP 1000, Step 0-5):")
    print(all_tx[1000][:6].tolist())

    print("-" * 30)
    print("3️⃣ 最后 5 个交易 (EP 4999, Step 995-999):")
    print(all_tx[-1][-5:].tolist())
    print("=" * 50)

    # 顺手存一个很简短的说明文件
    summary_path = report_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Uniform tx pool\n")
        f.write(f"T={T}\n")
        f.write(f"Episodes={EPISODES}\n")
        f.write(f"Steps_per_episode={STEPS_PER_EP}\n")
        f.write(f"Seed={SEED}\n")
        f.write(f"Shape={all_tx.shape}\n")
        f.write(f"EP0_first5={all_tx[0][:5].tolist()}\n")
        f.write(f"EP1000_first6={all_tx[1000][:6].tolist()}\n")
        f.write(f"EP4999_last5={all_tx[-1][-5:].tolist()}\n")

    print(f"\n📝 简要说明已保存: {summary_path}")


if __name__ == "__main__":
    generate_tx_pool()