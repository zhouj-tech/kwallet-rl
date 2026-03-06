import numpy as np

# 这里的参数必须和你的 CONFIG 保持一致
EPISODES = 5000       # 预生成的总回合数
STEPS_PER_EP = 1000   # 每个回合的步数
MAX_TX = 1000         # T 值 (最大交易金额)
SEED = 123            # 统一种子

def generate_shared_data():
    # 1. 使用独立的随机数生成器
    rng = np.random.default_rng(SEED)
    
    # 2. 生成大矩阵 (5000, 1000)
    all_tx = rng.integers(1, MAX_TX + 1, size=(EPISODES, STEPS_PER_EP))
    
    # 3. 保存文件
    np.save("shared_tx_pool.npy", all_tx)
    print(f"✅ 成功！已生成 {EPISODES} 回合交易序列 -> shared_tx_pool.npy\n")

    # 4. 打印特定位置的交易额用于组员对齐
    print("="*50)
    print("📊 交易流对齐指纹 (用于组员核对)")
    print("="*50)
    
    # 前 5 个交易 (第 0 回合的前 5 步)
    print(f"1. 最初 5 个交易 (EP 0, Step 0-4):")
    print(f"   {all_tx[0][:5].tolist()}")
    print("-" * 30)

    # 中间第 1000 回合的前 6 步
    print(f"2. 中间交易 (EP 1000, Step 0-5):")
    print(f"   {all_tx[1000][:6].tolist()}")
    print("-" * 30)

    # 最后 5 个交易 (最后一回合的最后 5 步)
    print(f"3. 最后 5 个交易 (EP 4999, Step 995-999):")
    print(f"   {all_tx[-1][-5:].tolist()}")
    print("="*50)
    

if __name__ == "__main__":
    generate_shared_data()









    