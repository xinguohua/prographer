# =============================================================================
#  修改后的 ProGrapher 训练模块 (train_model.py)
#  版本：兼容旧的函数调用，无需修改 train.py
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict


# =============================================================================
#  1. 定义 ProGrapher 的异常检测器 (Anomaly Detector)
#     根据论文4.3节，使用 TextRCNN 模型结构
# =============================================================================
class AnomalyDetector(nn.Module):
    """
    使用 TextRCNN 结构实现的异常检测器，以匹配 ProGrapher 论文。
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers, dropout, kernel_sizes, num_filters):
        super(AnomalyDetector, self).__init__()

        self.lstm = nn.LSTM( # 定义一个 LSTM (长短期记忆网络) 层，这是处理序列数据的核心。
            input_size=embedding_dim, #告诉LSTM，输入序列中每个元素（每个快照嵌入）的向量维度是多少。
            hidden_size=hidden_dim, # LSTM内部“记忆单元”的维度。
            num_layers=num_layers, #叠的LSTM层数。多层网络能学习更复杂的模式。
            batch_first=True,   #规定输入数据的维度顺序为 (批次大小, 序列长度, 特征维度)，这更符合直觉。
            bidirectional=True, #创建了一个双向LSTM。它会同时从前向后和从后向前读取序列。这样做的好处是，在任何一个时间点，模型的输出都同时包含了该点之前和之后的上下文信息，理解更全面
            dropout=dropout if num_layers > 1 else 0 #如果有多层LSTM，就在层与层之间随机丢弃一些连接，防止过拟合。
        )

        self.convs = nn.ModuleList([ #定义一个 ModuleList，它像一个Python列表，但能让PyTorch正确地管理里面的所有层。
            nn.Conv1d(in_channels=hidden_dim * 2, #卷积层的输入通道数
                      out_channels=num_filters, #卷积层输出的通道数
                      kernel_size=k) #卷积核的大小
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(dropout) #定义一个 Dropout 层
        self.fc = nn.Linear(num_filters * len(kernel_sizes), embedding_dim)  #定义一个 全连接（线性）层 (nn.Linear)

    def forward(self, sequence_embeddings):
        lstm_out, _ = self.lstm(sequence_embeddings) #首先通过双向LSTM层。
        conv_in = lstm_out.permute(0, 2, 1) #permute 用于交换张量的维度。

        # 优化后的卷积和池化操作
        pooled_outputs = []
        for conv in self.convs: #遍历我们创建的每一个卷积层
            conv_out = F.relu(conv(conv_in))  #将 conv_in 输入到卷积层，然后用 F.relu 激活函数处理。
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.shape[2]).squeeze(2)  #最大池化
            pooled_outputs.append(pooled)  #将每个卷积核提取出的最强特征信号

        concatenated = torch.cat(pooled_outputs, dim=1)  #将所有来自不同尺寸卷积核的池化结果（特征向量）拼接成一个更长的向量。
        dropped_out = self.dropout(concatenated)   #将拼接后的特征向量通过 Dropout 层。
        predicted_embedding = self.fc(dropped_out) #将处理后的特征向量输入到最后的全连接层，得到最终的预测结果——一个维度为 embedding_dim 的向量。

        return predicted_embedding

# =============================================================================
#  2. 训练模型的主函数 (兼容旧的调用签名)
# =============================================================================
def train_model(
        snapshot_embeddings,
        sequence_length_L=12,
        embedding_dim=256,
        hidden_dim=128,
        num_layers=5,
        dropout_rate=0.2,
        kernel_sizes=[3, 4, 5],
        num_filters=100,
        seq_lr=3e-4,
        seq_epochs=30,
        batch_size=32,
        model_save_path="prographer_detector.pth"
):
    """
    ProGrapher 的异常检测器训练 (TextRCNN 版本)。
    训练完成后只保存最优模型参数 (state_dict)，返回模型和训练历史。
    """

    print(f"训练使用序列长度: {sequence_length_L}")
    print(f"模型参数: 嵌入维度={embedding_dim}, 隐藏维度={hidden_dim}")
    print(f"LSTM层数={num_layers}, Dropout={dropout_rate}")
    print(f"卷积核大小={kernel_sizes}, 滤波器数={num_filters}")
    print("-" * 50)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- ProGrapher 训练后端运行在: {device} ---")

    # ========== 1. 数据准备 ==========
    snapshot_embeddings_tensor = torch.tensor(snapshot_embeddings, dtype=torch.float32)

    if snapshot_embeddings_tensor.size(0) <= sequence_length_L:
        raise ValueError(f"快照数量 {len(snapshot_embeddings_tensor)} 不足以构成一个长度为 {sequence_length_L} 的序列")

    def make_windows(x, L):
        seqs, tars = [], []
        for i in range(len(x) - L):
            seqs.append(x[i: i + L])
            tars.append(x[i + L])
        return torch.stack(seqs), torch.stack(tars)

    split_idx = int(0.8 * len(snapshot_embeddings_tensor))
    train_arr, val_arr = snapshot_embeddings_tensor[:split_idx], snapshot_embeddings_tensor[split_idx:]

    train_x, train_y = make_windows(train_arr, sequence_length_L)
    val_x, val_y = make_windows(val_arr, sequence_length_L)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    print(f"训练集样本数: {len(train_x)}, 验证集样本数: {len(val_x)}")

    # ========== 2. 初始化模型、优化器和损失函数 ==========
    detector_model = AnomalyDetector(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout_rate,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters
    ).to(device)

    optimizer = optim.Adam(detector_model.parameters(), lr=seq_lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                     factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    # ========== 3. 训练循环 ==========
    best_val_loss = float("inf")
    best_state = OrderedDict((k, v.cpu()) for k, v in detector_model.state_dict().items())
    patience, bad_epochs = 5, 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(seq_epochs):
        # ---- 训练 ----
        detector_model.train()
        train_loss = 0.0
        for seq_batch, target_batch in tqdm(train_loader, desc=f"Train {epoch+1}/{seq_epochs}", leave=False):
            seq_batch, target_batch = seq_batch.to(device), target_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = detector_model(seq_batch)
                loss = criterion(pred, target_batch)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(detector_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * seq_batch.size(0)

        train_loss /= len(train_x)

        # ---- 验证 ----
        detector_model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            for seq_batch, target_batch in val_loader:
                seq_batch, target_batch = seq_batch.to(device), target_batch.to(device)
                pred = detector_model(seq_batch)
                vloss = criterion(pred, target_batch)
                val_loss += vloss.item() * seq_batch.size(0)

        val_loss /= len(val_x)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{seq_epochs} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # ---- 保存最佳权重 ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in detector_model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("早停触发，停止训练。")
                break

    # ========== 4. 保存模型 ==========
    if best_state is not None:
        detector_model.load_state_dict(best_state)

    if model_save_path:
        save_dir = os.path.dirname(model_save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 只保存模型参数
        torch.save(detector_model.state_dict(), model_save_path)
        print(f"最优模型参数已保存至: {os.path.abspath(model_save_path)}")

    return detector_model, history