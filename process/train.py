# =================训练=========================
import sys
import os
# 这个代码块修复了导入路径问题
# 它将父目录（也就是项目根目录）添加到了 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.match.match import train_model
import platform
import yaml

from process.partition import detect_communities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【新增】可自定义的训练参数
SEQUENCE_LENGTH = 12        # 序列长度，可以修改


print(f"训练参数: 序列长度={SEQUENCE_LENGTH}")

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

system = platform.system().lower()
if "windows" in system:
    env_config = config["local"]
else:
    env_config = config["remote"]

PATH_MAP = env_config["path_map"]


# 获取数据集
data_handler = get_handler("atlas", True, PATH_MAP)
# data_handler = get_handler("theia", True)

# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
data_handler.build_graph()
G_snapshots = data_handler.snapshots
print(f"总共生成了 {len(G_snapshots)} 个快照。")

#嵌入构造特征向量
embedder_class = get_embedder_by_name("prographer")
# 【修改】传递序列长度参数到embedder
embedder = embedder_class(G_snapshots, sequence_length=SEQUENCE_LENGTH)
embedder.train()
snapshot_embeddings = embedder.get_snapshot_embeddings()
rsg_embeddings, rsg_vocab = embedder.get_rsg_embeddings()
print("\n--- Encoder process finished ---")
print(f"已生成快照嵌入序列，形状为: {snapshot_embeddings.shape}")
print(f"已生成RSG嵌入矩阵，形状为: {rsg_embeddings.shape}")
print(f"RSG词汇表大小: {len(rsg_vocab)}")
# 模型训练

train_model(snapshot_embeddings[data_handler.benign_idx_start:data_handler.benign_idx_end+1], sequence_length_L=SEQUENCE_LENGTH)