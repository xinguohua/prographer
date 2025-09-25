# =================训练=========================
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.match.match import train_model
from process.partition import create_snapshots_from_separate_data
import platform
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【新增】可自定义的训练参数
SEQUENCE_LENGTH = 12       # 序列长度，可以修改

print(f"训练参数: 序列长度={SEQUENCE_LENGTH}")

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

system = platform.system().lower()
if "windows" in system:
    env_config = config["local"]
else:
    env_config = config["remote"]

PATH_MAP = env_config["path_map"]

# ======================== 阶段1：编码器训练（使用全部数据）========================
print("="*60)
print("阶段1：编码器训练（使用全部数据）")
print("="*60)

# 获取数据集用于编码器训练（加载全部数据）
encoder_handler = get_handler("cadets", True, PATH_MAP)  # train=True，但会加载全部数据
encoder_handler.load(load_all_for_encoder=True)  # 使用新参数加载全部数据

# 使用全部数据创建快照
all_snapshots, benign_start, benign_end, malicious_start, malicious_end = create_snapshots_from_separate_data(encoder_handler)

# 设置编码器handler的属性
encoder_handler.snapshots = all_snapshots
encoder_handler.benign_idx_start = benign_start
encoder_handler.benign_idx_end = benign_end
encoder_handler.malicious_idx_start = malicious_start
encoder_handler.malicious_idx_end = malicious_end

print(f"编码器训练数据统计:")
print(f"良性快照索引范围: {encoder_handler.benign_idx_start} 到 {encoder_handler.benign_idx_end}")
print(f"恶意快照索引范围: {encoder_handler.malicious_idx_start} 到 {encoder_handler.malicious_idx_end}")
print(f"良性快照数量: {benign_end - benign_start + 1 if benign_start != -1 else 0}")
print(f"恶意快照数量: {malicious_end - malicious_start + 1 if malicious_start != -1 else 0}")
print(f"总共生成了 {len(all_snapshots)} 个快照用于编码器训练")


with open("communities_all.txt", "w", encoding="utf-8") as f:
    for i, g in enumerate(all_snapshots):
        print(f"正在写社区 {i} ...")  # 打印进度
        f.write(f"Community {i}:\n")
        for v in g.vs:
            attrs = v.attributes()
            attr_str = ", ".join([f"{k}={v[k]}" for k in attrs])
            f.write(f"  Vertex {v.index}: {attr_str}\n")
        f.write("\n")

# 🔥 保存快照数据到文件
print("\n--- 保存快照数据到文件 ---")
import pickle
snapshot_data = {
    'all_snapshots': all_snapshots,
    'benign_idx_start': benign_start,
    'benign_idx_end': benign_end,
    'malicious_idx_start': malicious_start,
    'malicious_idx_end': malicious_end,
}

snapshot_file = "snapshot_data.pkl"
with open(snapshot_file, 'wb') as f:
    pickle.dump(snapshot_data, f)

print(f"✅ 快照数据已保存到: {snapshot_file}")
print(f"  - 总快照数: {len(all_snapshots)}")
print(f"  - 良性快照范围: {benign_start} 到 {benign_end}")
print(f"  - 恶意快照范围: {malicious_start} 到 {malicious_end}")

# 使用全部快照训练编码器
print("\n--- 编码器训练（全部数据）---")
embedder_class = get_embedder_by_name("prographer")
embedder = embedder_class(all_snapshots, sequence_length=SEQUENCE_LENGTH)
embedder.train()  # 在全部快照上训练编码器
all_snapshot_embeddings = embedder.get_snapshot_embeddings()
rsg_embeddings, rsg_vocab = embedder.get_rsg_embeddings()

print("\n--- Encoder process finished ---")
print(f"已生成快照嵌入序列，形状为: {all_snapshot_embeddings.shape}")
print(f"已生成RSG嵌入矩阵，形状为: {rsg_embeddings.shape}")
print(f"RSG词汇表大小: {len(rsg_vocab)}")

# ======================== 阶段2：异常检测器训练（只用良性数据）========================
print("\n" + "="*60)
print("阶段2：异常检测器训练（只用良性数据）")
print("="*60)

# 提取良性快照的嵌入用于异常检测器训练
if encoder_handler.benign_idx_start != -1:
    benign_embeddings = all_snapshot_embeddings[encoder_handler.benign_idx_start:encoder_handler.benign_idx_end+1]
    
    print(f"异常检测器训练数据统计:")
    print(f"用于异常检测器训练的良性嵌入形状: {benign_embeddings.shape}")
    print(f"良性快照索引范围: {encoder_handler.benign_idx_start} 到 {encoder_handler.benign_idx_end}")
    
    # 训练异常检测器（只用良性嵌入）
    train_model(benign_embeddings, sequence_length_L=SEQUENCE_LENGTH)
    
    print(f"\n✅ 训练完成！")
    print(f"  - 编码器使用了 {len(all_snapshots)} 个快照（良性+恶意）")
    print(f"  - 异常检测器使用了 {len(benign_embeddings)} 个良性快照")
else:
    print("❌ 错误：没有良性快照可用于训练")