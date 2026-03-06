import numpy as np
import os



T = 100                # 最大交易金额
EPISODES = 5000        # 生成的回合数
STEPS_PER_EP = 1000    # 每回合交易数量
SEED = 123             # 随机种子


def generate_tx_pool():

    print("=" * 60)
    print("🚀 开始生成交易序列 (T=100)")
    print("=" * 60)

    print(f"T (最大交易金额): {T}")
    print(f"Episodes: {EPISODES}")
    print(f"Steps per episode: {STEPS_PER_EP}")
    print(f"Seed: {SEED}")

    # 创建 data 文件夹
    os.makedirs("data", exist_ok=True)

    filename = f"tx_pool_T{T}.npy"
    save_path = os.path.join("data", filename)

    # =====================================================
    # 1. 创建随机数生成器
    # =====================================================

    rng = np.random.default_rng(SEED)

    # =====================================================
    # 2. 生成交易矩阵
    # =====================================================

    all_tx = rng.integers(
        1,
        T + 1,
        size=(EPISODES, STEPS_PER_EP)
    )

    # =====================================================
    # 3. 保存数据
    # =====================================================

    np.save(save_path, all_tx)

    print("\n✅ 生成完成")
    print(f"📁 文件路径: {save_path}")
    print(f"📊 矩阵形状: {all_tx.shape}")

    # =====================================================
    # 4. 打印数据指纹（用于组员核对）
    # =====================================================

    print("\n" + "=" * 50)
    print("📊 交易流对齐指纹 (用于组员核对)")
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

    print("\n🎯 如果组员使用相同 seed 和参数，这些数字必须完全一致。")


if __name__ == "__main__":
    generate_tx_pool()