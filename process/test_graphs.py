import os
import sys
import yaml
import textwrap  # 导入 textwrap 以便格式化长文本
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# --- 1. 设置和导入 ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process.datahandlers import get_handler
from process.embedders import ProGrapherEmbedder

# --- 2. 模型定义 (不变) ---
class AnomalyDetector(nn.Module):
    # ... 此处代码无变化 ...
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, kernel_sizes, num_filters):
        super(AnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=hidden_dim * 2, out_channels=num_filters, kernel_size=k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embedding_dim)
    def forward(self, sequence_embeddings):
        lstm_out, _ = self.lstm(sequence_embeddings)
        conv_in = lstm_out.permute(0, 2, 1)
        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(conv_in))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)
        concatenated = torch.cat(pooled_outputs, dim=1)
        dropped_out = self.dropout(concatenated)
        predicted_embedding = self.fc(dropped_out)
        return predicted_embedding

# --- 3. 超参数设置 (可通过参数自定义) ---
SEQUENCE_LENGTH_L = 12
EMBEDDING_DIM = 256
HIDDEN_DIM = 128
NUM_LAYERS = 5
DROPOUT_RATE = 0.2
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100
DETECTION_THRESHOLD = 0.016
TOP_K_INDICATORS = 5
WL_DEPTH = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --- 4. 辅助函数 (不变) ---
def save_snapshot_nodes_to_file(all_snapshots, output_dir="d:/prographer/process"):
    """将每个快照中的节点信息保存到txt文件"""
    import os
    from datetime import datetime
    
    # 创建输出文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"snapshot_nodes_{timestamp}.txt")
    
    print(f"\n--- 保存快照节点信息到文件 ---")
    print(f"输出文件: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== ProGrapher 快照节点详情报告 ===\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总快照数: {len(all_snapshots)}\n")
            f.write("=" * 60 + "\n\n")
            
            for i, snapshot in enumerate(all_snapshots):
                f.write(f"快照 {i}:\n")
                f.write(f"  节点总数: {len(snapshot.vs)}\n")
                f.write(f"  边总数: {len(snapshot.es)}\n")
                f.write("  节点详情:\n")
                
                # 统计节点类型
                node_type_count = {}
                malicious_count = 0
                
                for v in snapshot.vs:
                    node_name = v['name']
                    node_type = v.attributes().get('type_name', 'UNKNOWN')
                    label = v.attributes().get('label', 0)
                    
                    # 统计节点类型
                    node_type_count[node_type] = node_type_count.get(node_type, 0) + 1
                    
                    # 统计恶意节点
                    if label == 1:
                        malicious_count += 1
                    
                    # 写入节点信息
                    status = "🔴恶意" if label == 1 else "🟢正常"
                    f.write(f"    {node_name} | 类型:{node_type} | 状态:{status}\n")
                
                # 写入统计信息
                f.write(f"  统计信息:\n")
                f.write(f"    恶意节点数: {malicious_count}/{len(snapshot.vs)}\n")
                f.write(f"    节点类型分布:\n")
                for node_type, count in sorted(node_type_count.items()):
                    f.write(f"      {node_type}: {count}个\n")
                f.write("\n" + "-" * 50 + "\n\n")
        
        print(f"✅ 快照节点信息已保存到: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"❌ 保存快照节点信息时出错: {str(e)}")
        return None

def predict_anomalous_snapshots(snapshot_embeddings, model_path, dynamic_sequence_length=None):
    # 【修改】动态调整卷积核大小以适应序列长度
    effective_sequence_length = dynamic_sequence_length if dynamic_sequence_length is not None else SEQUENCE_LENGTH_L
    
    # 【新增】根据序列长度动态调整卷积核，确保不会出错
    max_kernel_size = max(KERNEL_SIZES)
    if effective_sequence_length < max_kernel_size:
        # 如果序列长度小于最大卷积核，调整卷积核大小
        adjusted_kernel_sizes = [k for k in KERNEL_SIZES if k <= effective_sequence_length]
        if not adjusted_kernel_sizes:
            adjusted_kernel_sizes = [1]  # 至少保持一个卷积核
        print(f"警告: 序列长度({effective_sequence_length})小于最大卷积核({max_kernel_size})")
        print(f"调整卷积核从 {KERNEL_SIZES} 到 {adjusted_kernel_sizes}")
        kernel_sizes_to_use = adjusted_kernel_sizes
    else:
        kernel_sizes_to_use = KERNEL_SIZES
    
    detector_model = AnomalyDetector(
        embedding_dim=EMBEDDING_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        dropout=DROPOUT_RATE, 
        kernel_sizes=kernel_sizes_to_use,  # 使用调整后的卷积核
        num_filters=NUM_FILTERS
    ).to(device)
    
    detector_model.load_state_dict(torch.load(model_path, map_location=device))
    detector_model.eval()
    tensor = torch.tensor(snapshot_embeddings, dtype=torch.float32)
    
    # 【修改】为所有快照初始化预测标签
    snapshot_pred_labels = np.zeros(len(tensor), dtype=int)
    diff_vectors = {}
    snapshot_scores = {}  # 【新增】存储每个快照的得分
    
    print(f"快照嵌入张量形状: {tensor.shape}")
    print(f"预测标签数组初始长度: {len(snapshot_pred_labels)}")
    print(f"使用序列长度: {effective_sequence_length}")
    print(f"使用卷积核大小: {kernel_sizes_to_use}")
    
    total_snapshots = len(tensor)
    
    with torch.no_grad():
        # 【关键修改】为了检测所有快照（包括第一个），我们需要不同的策略
        for i in tqdm(range(total_snapshots), desc="检测快照序列", leave=True, unit="snapshot"):
            
            if i < effective_sequence_length:
                # 【新策略】对于前面的快照，使用零填充或重复填充
                if i == 0:
                    # 第一个快照：使用当前快照重复填充
                    sequence = tensor[0].unsqueeze(0).repeat(effective_sequence_length, 1)
                else:
                    # 前面几个快照：使用零填充 + 可用快照
                    padding_size = effective_sequence_length - i
                    padding = torch.zeros(padding_size, tensor.shape[1])
                    available_snapshots = tensor[:i]
                    sequence = torch.cat([padding, available_snapshots], dim=0)
                
                target = tensor[i]
            else:
                # 【标准策略】使用滑动窗口
                sequence = tensor[i-effective_sequence_length:i]
                target = tensor[i]
            
            # 预测和检测
            sequence = sequence.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            prediction = detector_model(sequence).squeeze(0)
            error = torch.nn.functional.mse_loss(prediction, target).item()
            diff_vector = (prediction - target).cpu().numpy()
            
            # 【新增】记录每个快照的得分
            snapshot_scores[i] = error
            
            # 标记异常
            if error > DETECTION_THRESHOLD:
                snapshot_pred_labels[i] = 1
                diff_vectors[i] = {
                    "position": i, 
                    "error": error, 
                    "diff_vector": diff_vector, 
                    "real_embedding": target.cpu().numpy(), 
                    "pred_embedding": prediction.cpu().numpy()
                }
    
    # 【新增】打印每个快照的得分
    print(f"\n--- 快照得分详情 ---")
    print(f"检测阈值: {DETECTION_THRESHOLD}")
    print("快照索引 | 得分(Error) | 状态")
    print("-" * 40)
    for i in range(total_snapshots):
        score = snapshot_scores[i]
        status = "🔴 异常" if score > DETECTION_THRESHOLD else "🟢 正常"
        print(f"快照 {i:2d}   | {score:.6f}   | {status}")
    print("-" * 40)
    
    return snapshot_pred_labels, diff_vectors

def get_true_snapshot_labels(snapshots):
    # ... 此函数内容不变 ...
    true_labels = []
    for snapshot in snapshots:
        malicious_nodes = any(v['label'] == 1 for v in snapshot.vs)
        true_labels.append(1 if malicious_nodes else 0)
    return np.array(true_labels)

def generate_key_indicators(all_snapshots, diff_vectors, rsg_embeddings, rsg_vocab):
    # ... 此函数内容不变 ...
    print("\n" + "="*50); print(" 关键指标生成器 - 异常RSG排名"); print("="*50)
    if not diff_vectors: print("未检测到任何异常快照，跳过指标生成"); return
    progress = tqdm(diff_vectors.items(), total=len(diff_vectors), desc="分析异常快照")
    for idx, anomaly_info in progress:
        # 【修复】检查索引是否在快照范围内
        if idx >= len(all_snapshots):
            print(f"\n警告: 异常快照索引 {idx} 超出快照范围 (0-{len(all_snapshots)-1})，跳过")
            continue
            
        snapshot = all_snapshots[idx]; diff_vector = anomaly_info["diff_vector"]
        if diff_vector.ndim > 1: diff_vector = diff_vector.squeeze()
        rsg_scores = defaultdict(float); rsg_count = 0
        for v_idx in range(len(snapshot.vs)):
            for d in range(WL_DEPTH + 1):
                rsg_str = ProGrapherEmbedder.generate_rsg(snapshot, v_idx, d)
                if rsg_str in rsg_vocab:
                    rsg_id = rsg_vocab[rsg_str]; rsg_vec = rsg_embeddings[rsg_id]
                    if rsg_vec.ndim > 1: rsg_vec = rsg_vec.squeeze()
                    score = np.abs(np.vdot(diff_vector, rsg_vec)); rsg_scores[rsg_str] = max(rsg_scores[rsg_str], score); rsg_count += 1
        if not rsg_scores: progress.set_postfix(snapshot=f"{idx}", info=f"未找到RSG"); continue
        sorted_rsgs = sorted(rsg_scores.items(), key=lambda x: x[1], reverse=True)
        progress.set_postfix(snapshot=f"{idx}", score=f"{sorted_rsgs[0][1]:.4f}")
        print(f"\n异常快照 {idx} (检测差异: {anomaly_info['error']:.6f})"); print(f"  - 总RSG数量: {rsg_count}"); print(f"  - Top-{TOP_K_INDICATORS} 可疑 RSG:")
        for i, (rsg, score) in enumerate(sorted_rsgs[:TOP_K_INDICATORS]): print(f"    {i+1}. {rsg} (可疑度: {score:.6f})")
    print("="*50 + "\n")

# =========================================================================
# =================== 增强的调试信息函数 ==================================
# =========================================================================
def print_debug_info(all_snapshots, eval_true, eval_pred, eval_start_idx):
    """
    详细打印TP、FP、FN、TN快照的调试信息，显示导致分类的具体节点。
    """
    print("\n" + "="*70)
    print(" 🔍 详细调试信息 (TP / FP / FN / TN 完整分析)")
    print("="*70)

    # 分类收集各种情况的快照索引
    tp_indices = []  # 真阳性：真实恶意 + 预测恶意
    fp_indices = []  # 假阳性：真实良性 + 预测恶意
    fn_indices = []  # 假阴性：真实恶意 + 预测良性
    tn_indices = []  # 真阴性：真实良性 + 预测良性

    for i in range(len(eval_true)):
        snapshot_idx = i + eval_start_idx
        true_label = eval_true[i]
        pred_label = eval_pred[i]

        if true_label == 1 and pred_label == 1:
            tp_indices.append(snapshot_idx)
        elif true_label == 0 and pred_label == 1:
            fp_indices.append(snapshot_idx)
        elif true_label == 1 and pred_label == 0:
            fn_indices.append(snapshot_idx)
        else:  # true_label == 0 and pred_label == 0
            tn_indices.append(snapshot_idx)

    # 打印统计概览
    print(f"\n📊 快照分类统计:")
    print(f"  ✅ 真阳性 (TP): {len(tp_indices)} 个快照")
    print(f"  ❌ 假阳性 (FP): {len(fp_indices)} 个快照")
    print(f"  ⚠️  假阴性 (FN): {len(fn_indices)} 个快照")
    print(f"  ✓  真阴性 (TN): {len(tn_indices)} 个快照")

    # === 详细分析 TP 快照 ===
    if tp_indices:
        print("\n" + "="*50)
        print("✅ 真阳性 (TP) 快照详细分析 - 正确检测到的恶意快照")
        print("="*50)
        for snapshot_idx in tp_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n🎯 快照 {snapshot_idx}:")

            # 分析节点类型
            malicious_nodes = []
            benign_nodes = []
            node_types_count = {}

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1

                if v.attributes().get('label') == 1:
                    malicious_nodes.append(f"{node_name}({node_type})")
                else:
                    benign_nodes.append(f"{node_name}({node_type})")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  🔴 恶意节点 ({len(malicious_nodes)}个):")
            if malicious_nodes:
                malicious_str = ', '.join(malicious_nodes[:10])  # 最多显示10个
                if len(malicious_nodes) > 10:
                    malicious_str += f" ... (+{len(malicious_nodes)-10}个更多)"
                print(f"      {malicious_str}")

            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")

    # === 详细分析 FP 快照 ===
    if fp_indices:
        print("\n" + "="*50)
        print("❌ 假阳性 (FP) 快照详细分析 - 误报的良性快照")
        print("="*50)
        for snapshot_idx in fp_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n🚨 快照 {snapshot_idx} (误报):")

            # 分析节点类型分布，寻找误报原因
            node_types_count = {}
            suspicious_patterns = []
            all_nodes = []

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1
                all_nodes.append(f"{node_name}({node_type})")

                # 检查可能导致误报的模式
                if 'SUBJECT_PROCESS' in node_type and any(word in node_name.lower()
                                                          for word in ['system', 'admin', 'service', 'daemon']):
                    suspicious_patterns.append(f"系统进程: {node_name}")
                elif 'NETFLOW' in node_type:
                    suspicious_patterns.append(f"网络流: {node_name}")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")

            if suspicious_patterns:
                print(f"  ⚡ 可能的误报原因:")
                for pattern in suspicious_patterns[:5]:  # 最多显示5个
                    print(f"      • {pattern}")

            # 显示部分节点名称用于分析
            print(f"  📝 部分节点 (前10个):")
            sample_nodes = ', '.join(all_nodes[:10])
            if len(all_nodes) > 10:
                sample_nodes += f" ... (+{len(all_nodes)-10}个更多)"
            wrapped_nodes = textwrap.fill(sample_nodes, width=70, initial_indent='      ', subsequent_indent='      ')
            print(wrapped_nodes)

    # === 详细分析 FN 快照 ===
    if fn_indices:
        print("\n" + "="*50)
        print("⚠️ 假阴性 (FN) 快照详细分析 - 漏检的恶意快照")
        print("="*50)
        for snapshot_idx in fn_indices:
            snapshot = all_snapshots[snapshot_idx]
            print(f"\n⚠️  快照 {snapshot_idx} (漏检):")

            # 分析为什么这些恶意节点没被检测到
            malicious_nodes = []
            benign_nodes = []
            node_types_count = {}

            for v in snapshot.vs:
                node_name = v['name']
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                node_types_count[node_type] = node_types_count.get(node_type, 0) + 1

                if v.attributes().get('label') == 1:
                    malicious_nodes.append(f"{node_name}({node_type})")
                else:
                    benign_nodes.append(f"{node_name}({node_type})")

            print(f"  📈 总节点数: {len(snapshot.vs)}, 边数: {len(snapshot.es)}")
            print(f"  🔴 被漏检的恶意节点 ({len(malicious_nodes)}个):")
            if malicious_nodes:
                malicious_str = ', '.join(malicious_nodes)
                wrapped_malicious = textwrap.fill(malicious_str, width=70, initial_indent='      ', subsequent_indent='      ')
                print(wrapped_malicious)

            print(f"  📊 节点类型分布: {dict(sorted(node_types_count.items()))}")
            print(f"  💡 可能的漏检原因: 恶意节点比例较低 ({len(malicious_nodes)}/{len(snapshot.vs)} = {len(malicious_nodes)/len(snapshot.vs)*100:.1f}%)")

    # === 简要显示 TN 快照统计 ===
    if tn_indices:
        print("\n" + "="*50)
        print("✓ 真阴性 (TN) 快照统计 - 正确识别的良性快照")
        print("="*50)
        print(f"  ✅ 共有 {len(tn_indices)} 个快照被正确识别为良性")

        # 统计TN快照的节点类型分布
        if len(tn_indices) > 0:
            sample_tn = all_snapshots[tn_indices[0]]  # 取一个样本
            tn_node_types = {}
            for v in sample_tn.vs:
                node_type = v.attributes().get('type_name', 'UNKNOWN')
                tn_node_types[node_type] = tn_node_types.get(node_type, 0) + 1
            print(f"  📊 典型良性快照的节点类型分布 (快照{tn_indices[0]}): {dict(sorted(tn_node_types.items()))}")

    print("\n" + "="*70)
    print("🎯 调试分析总结:")
    print(f"  • 总共分析了 {len(eval_true)} 个快照")
    print(f"  • 检测准确率: {(len(tp_indices) + len(tn_indices))/len(eval_true)*100:.1f}%")
    if len(tp_indices) + len(fn_indices) > 0:
        print(f"  • 恶意快照召回率: {len(tp_indices)/(len(tp_indices) + len(fn_indices))*100:.1f}%")
    if len(tp_indices) + len(fp_indices) > 0:
        print(f"  • 恶意检测精确率: {len(tp_indices)/(len(tp_indices) + len(fp_indices))*100:.1f}%")
    print("="*70)

# =========================================================================
# =================== 核心评估函数 (保持不变) ===========================
# =========================================================================
def run_snapshot_level_evaluation(detector_model_path, encoder_model_path, PATH_MAP, MALICIOUS_INTERVALS_PATH,
                                  sequence_length=12,
                                  test_window_minutes=20,
                                  ):
    """运行快照级别的异常检测评估 - 评估所有快照"""
    handler = get_handler(
        "atlas", 
        False,
        PATH_MAP,
        MALICIOUS_INTERVALS_PATH,
        use_time_split=True,
        test_window_minutes=test_window_minutes
    )
    handler.load()
    all_snapshots,complete_nodes_per_graph, labels_per_graph = handler.build_graph()
    if not all_snapshots: 
        print("错误: 未能构建任何快照。")
        return
        
    # 保存快照节点信息到文件
    save_snapshot_nodes_to_file(all_snapshots)
    
    true_labels = get_true_snapshot_labels(all_snapshots)
    
    print(f"\n--- 调试信息 ---")
    print(f"总快照数: {len(all_snapshots)}")
    print(f"真实标签数: {len(true_labels)}")
    print(f"真实标签内容: {true_labels}")

    print(f"✅ 将评估所有 {len(all_snapshots)} 个快照")
    
    print("\n--- 加载预训练的编码器 ---")
    embedder = ProGrapherEmbedder.load(encoder_model_path, snapshot_sequence=all_snapshots)
    snapshot_embeddings = embedder.get_snapshot_embeddings()
    rsg_embeddings, rsg_vocab = embedder.get_rsg_embeddings()
    print(f"RSG嵌入加载完毕，词汇大小: {len(rsg_vocab)}")
    
    # 【关键修改】无论快照数量多少，都进行异常检测
    pred_labels, diff_vectors = predict_anomalous_snapshots(
        snapshot_embeddings, detector_model_path
    )
    print(f"检测到 {len(diff_vectors)} 个异常快照")
    print(f"预测标签长度: {len(pred_labels)}")
    
    # 【关键修改】评估所有快照，从索引0开始
    eval_true = true_labels
    eval_pred = pred_labels
    
    print(f"✅ 评估所有快照: 0 到 {len(all_snapshots)-1}")
    print(f"评估真实标签长度: {len(eval_true)}")
    print(f"评估预测标签长度: {len(eval_pred)}")
    
    # 确保两个数组长度一致
    min_len = min(len(eval_true), len(eval_pred))
    if min_len == 0:
        print("错误: 没有足够的数据进行评估")
        return
    
    eval_true = eval_true[:min_len]
    eval_pred = eval_pred[:min_len]
    
    # 计算性能指标
    tp = np.sum((eval_true == 1) & (eval_pred == 1))
    fp = np.sum((eval_true == 0) & (eval_pred == 1))
    tn = np.sum((eval_true == 0) & (eval_pred == 0))
    fn = np.sum((eval_true == 1) & (eval_pred == 0))
    
    acc = accuracy_score(eval_true, eval_pred)
    prec = precision_score(eval_true, eval_pred, zero_division=0)
    rec = recall_score(eval_true, eval_pred, zero_division=0)
    f1 = f1_score(eval_true, eval_pred, zero_division=0)
    
    print("\n" + "="*50)
    print(" 快照级别评估结果 (所有快照)")
    print("="*50)
    print(f" 真阳性 (TP): {tp}")
    print(f" 假阳性 (FP): {fp}")
    print(f" 真阴性 (TN): {tn}")
    print(f" 假阴性 (FN): {fn}")
    print("\n 性能评分:")
    print(f" 准确率: {acc:.4f}")
    print(f" 精确率: {prec:.4f}")
    print(f" 召回率: {rec:.4f}")
    print(f" F1分数: {f1:.4f}")
    print("="*50)
    
    generate_key_indicators(all_snapshots, diff_vectors, rsg_embeddings, rsg_vocab)
    print_debug_info(all_snapshots, eval_true, eval_pred, 0)  # 从索引0开始

# --- 主程序入口 ---
if __name__ == '__main__':

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    system = platform.system().lower()
    if "windows" in system:
        env_config = config["local"]
    else:
        env_config = config["remote"]



    # 拿到路径
    DETECTOR_MODEL_PATH = env_config["DETECTOR_MODEL_PATH"]
    ENCODER_MODEL_PATH = env_config["ENCODER_MODEL_PATH"]
    MALICIOUS_INTERVALS_PATH = env_config["malicious_intervals"]
    PATH_MAP = env_config["path_map"]

    # 【新增】可自定义的参数
    SEQUENCE_LENGTH = 7        # 序列长度，可以修改
    TEST_WINDOW = 20           # 测试窗口分钟数，可以修改
    
    print(f"使用参数: 序列长度={SEQUENCE_LENGTH},  测试窗口={TEST_WINDOW}分钟")
    
    if not os.path.exists(DETECTOR_MODEL_PATH): print(f"错误: 检测器模型文件不存在: {DETECTOR_MODEL_PATH}"); sys.exit(1)
    if not os.path.exists(ENCODER_MODEL_PATH): print(f"错误: 编码器模型文件不存在: {ENCODER_MODEL_PATH}\n提示: 请先运行 train.py 来训练并生成编码器模型。"); sys.exit(1)
    
    run_snapshot_level_evaluation(DETECTOR_MODEL_PATH, ENCODER_MODEL_PATH, PATH_MAP, MALICIOUS_INTERVALS_PATH,
                                 SEQUENCE_LENGTH,  TEST_WINDOW)