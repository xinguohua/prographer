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
from process.partition import detect_communities

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【新增】可自定义的训练参数
SEQUENCE_LENGTH = 12        # 序列长度，可以修改
MALICIOUS_WINDOW = 10       # 恶意窗口分钟数，可以修改  
TEST_WINDOW = 20           # 测试窗口分钟数，可以修改

print(f"训练参数: 序列长度={SEQUENCE_LENGTH}, 恶意窗口={MALICIOUS_WINDOW}分钟, 测试窗口={TEST_WINDOW}分钟")

# 获取数据集
# 【修改】使用参数指定时间戳的数据分割
data_handler = get_handler("atlas", True, use_time_split=True, 
                          malicious_window_minutes=MALICIOUS_WINDOW, 
                          test_window_minutes=TEST_WINDOW)
# data_handler = get_handler("theia", True)

# 加载数据
data_handler.load()
# 成整个大图+捕捉特征语料+简化策略这里添加
# 【修改】更新返回值解包，接收新增的complete_nodes_per_graph和labels_per_graph
features, edges, mapp, relations, G_snapshots, snapshot_to_graph_map, graph_names_in_order, complete_nodes_per_graph, labels_per_graph = data_handler.build_graph()
print(f"总共生成了 {len(G_snapshots)} 个快照。")
#print("features:", features)
#print("edges:", edges)
#print("mapp:", mapp)
#print("relations:", relations)
#for i, snapshot in enumerate(G_snapshots):
# print(f"\n--- 快照 {i+1} ---")
# print(snapshot) # 单独打印每一个 snapshot 对象
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
# 匹配
# 【修改】传递序列长度参数到训练函数
train_model(G_snapshots, snapshot_embeddings, rsg_embeddings, rsg_vocab, 
           sequence_length_L=SEQUENCE_LENGTH)