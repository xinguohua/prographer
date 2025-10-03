import platform
import torch
import yaml

from datahandlers import get_handler
from embedders import get_embedder_by_name
from process.classfy import get_classfy

# ---------------- 配置参数 ----------------
CONFIG_PATH = "config.yaml"
DATASET_NAME = "atlas"          # 可切换数据集
# DATASET_NAME = "cadets"          # 可切换数据集
# EMBEDDER_NAME = "prographer"    # 嵌入器
EMBEDDER_NAME = "unicorn"    # 嵌入器
CLASSIFY_NAME = "prographer"     # 训练器
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict:
    """加载 YAML 配置，并根据系统环境选择配置分支"""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    system = platform.system().lower()
    return config["local"] if "windows" in system else config["remote"]


def prepare_data(path_map: dict):
    """加载数据并生成快照"""
    handler = get_handler(DATASET_NAME, True, path_map)
    handler.load()
    handler.build_graph()
    return handler


def build_embeddings(snapshots):
    """构建并训练嵌入器"""
    embedder_cls = get_embedder_by_name(EMBEDDER_NAME)
    embedder = embedder_cls(snapshots)
    embedder.train()
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    print("\n--- Encoder 过程完成 ---")
    print(f"[嵌入] 快照嵌入序列: {snapshot_embeddings.shape}")
    return snapshot_embeddings


def main():
    env_config = load_config(CONFIG_PATH)
    path_map = env_config["path_map"]

    # 数据准备
    handler = prepare_data(path_map)

    # 嵌入训练
    snapshot_embeddings = build_embeddings(handler.snapshots)

    # 模型训练
    benign_embeddings = snapshot_embeddings[handler.benign_idx_start:handler.benign_idx_end + 1]
    classify = get_classfy(CLASSIFY_NAME)
    classify.train(benign_embeddings)

if __name__ == "__main__":
    main()