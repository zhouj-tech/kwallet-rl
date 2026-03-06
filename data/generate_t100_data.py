import numpy as np

EPISODES = 5000       # 预生成的总回合数
STEPS_PER_EP = 1000   # 每个回合的步数
MAX_TX = 100          # 设定 T=100 (交易金额上限)
SEED = 123            # 统一种子保持不变

def generate_t100_data():
    # 1. 使用独立的随机数生成器
    rng = np.random.default_rng(SEED)
    
    # 2. 生成大矩阵 (5000, 1000)
    all_tx = rng.integers(1, MAX_TX + 1, size=(EPISODES, STEPS_PER_EP))
    
    # 3. 保存为新文件
    file_name = "shared_tx_pool_t100.npy"
    np.save(file_name, all_tx)
    print(f"✅ 成功！已生成 T=100 交易序列 -> {file_name}\n")

    # 4. 打印特定位置的交易额
    print("="*50)
    print("📊 T=100 交易流对齐指纹 (用于组员核对)")
    print("="*50)
    
    # 前 5 个交易
    print(f"1. 最初 5 个交易 (EP 0, Step 0-4):")
    print(f"   {all_tx[0][:5].tolist()}")
    print("-" * 30)

    # 中间第 1000 回合
    print(f"2. 中间交易 (EP 1000, Step 0-5):")
    print(f"   {all_tx[1000][:6].tolist()}")
    print("-" * 30)

    # 最后 5 个交易
    print(f"3. 最后 5 个交易 (EP 4999, Step 995-999):")
    print(f"   {all_tx[-1][-5:].tolist()}")
    print("="*50)
    
    print(f"💡 验证：当前序列最大值为 {np.max(all_tx)}，均值为 {np.mean(all_tx):.2f}")
    print(f"💡 在 k=6 (单桶 500) 的环境下，oversize_drops 预期将为 0。")

if __name__ == "__main__":
    generate_t100_data()