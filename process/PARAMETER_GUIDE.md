# ProGrapher 参数化序列长度实现说明

## 概述
已恢复到最初的参数化设计，您可以通过修改代码中的参数来自定义序列长度和其他配置。

## 修改的文件

### 1. `train.py`
- 添加了可自定义的参数变量
```python
SEQUENCE_LENGTH = 12        # 序列长度，可以修改
MALICIOUS_WINDOW = 10       # 恶意窗口分钟数，可以修改  
TEST_WINDOW = 20           # 测试窗口分钟数，可以修改
```
- 将序列长度参数传递给 embedder 和 train_model

### 2. `test_graphs.py`
- 添加了可自定义的参数变量
```python
SEQUENCE_LENGTH = 12        # 序列长度，可以修改
MALICIOUS_WINDOW = 10       # 恶意窗口分钟数，可以修改
TEST_WINDOW = 20           # 测试窗口分钟数，可以修改
```
- `run_snapshot_level_evaluation` 函数增加了序列长度等参数
- 移除了动态序列长度调整逻辑

### 3. `prographer_embedder.py`
- 构造函数增加了 `sequence_length` 参数
```python
def __init__(self, snapshot_sequence, ..., sequence_length=12):
```
- 在初始化时显示序列长度信息

### 4. `match.py`
- 恢复了原有的参数化设计
- `train_model` 函数保持所有参数可自定义

## 如何自定义序列长度

### 方法1: 修改代码中的变量
1. **训练时**: 修改 `train.py` 中的 `SEQUENCE_LENGTH = 12`
2. **测试时**: 修改 `test_graphs.py` 中的 `SEQUENCE_LENGTH = 12`

### 方法2: 在 prographer_embedder.py 中指定
您可以在创建 ProGrapherEmbedder 实例时传递序列长度：
```python
embedder = ProGrapherEmbedder(G_snapshots, sequence_length=16)
```

## 使用示例

### 训练 (序列长度=16)
```python
# 在 base.py 中修改
SEQUENCE_LENGTH = 16
```

### 测试 (序列长度=16)
```python  
# 在 test_graphs.py 中修改
SEQUENCE_LENGTH = 16
```

## 参数说明

- **SEQUENCE_LENGTH**: 用于训练和测试的序列长度
- **MALICIOUS_WINDOW**: 恶意事件周围的时间窗口（分钟）
- **TEST_WINDOW**: 测试数据的时间窗口（分钟）

## 验证一致性

运行时查看控制台输出：
- 训练阶段: "训练使用序列长度: X"
- 测试阶段: "使用序列长度: X (通过参数指定)"
- Embedder: "ProGrapherEmbedder sequence length: X"

确保所有地方显示的序列长度都一致。

## 注意事项

1. **确保一致性**: 训练和测试时必须使用相同的序列长度
2. **数据充足**: 序列长度不能超过可用快照数量
3. **模型兼容**: 更改序列长度后需要重新训练模型
