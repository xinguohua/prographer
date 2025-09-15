# 序列长度配置文件
# 确保训练和测试时使用相同的序列长度

# 核心参数：序列长度
SEQUENCE_LENGTH = 12

# 数据分割参数
MALICIOUS_WINDOW_MINUTES = 10
TEST_WINDOW_MINUTES = 20

# 模型参数
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.3
KERNEL_SIZES = [3, 4, 5]
NUM_FILTERS = 100

def get_sequence_length():
    """获取统一的序列长度"""
    return SEQUENCE_LENGTH

def get_data_split_params():
    """获取数据分割参数"""
    return {
        'malicious_window_minutes': MALICIOUS_WINDOW_MINUTES,
        'test_window_minutes': TEST_WINDOW_MINUTES
    }

def get_model_params():
    """获取模型参数"""
    return {
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT,
        'kernel_sizes': KERNEL_SIZES,
        'num_filters': NUM_FILTERS,
        'sequence_length': SEQUENCE_LENGTH
    }

def print_config():
    """打印当前配置"""
    print("=" * 50)
    print("ProGrapher 配置参数")
    print("=" * 50)
    print(f"序列长度: {SEQUENCE_LENGTH}")
    print(f"恶意事件窗口: ±{MALICIOUS_WINDOW_MINUTES} 分钟")
    print(f"测试窗口: ±{TEST_WINDOW_MINUTES} 分钟")
    print(f"嵌入维度: {EMBEDDING_DIM}")
    print(f"隐藏层维度: {HIDDEN_DIM}")
    print(f"LSTM层数: {NUM_LAYERS}")
    print(f"Dropout率: {DROPOUT}")
    print(f"卷积核大小: {KERNEL_SIZES}")
    print(f"卷积滤波器数: {NUM_FILTERS}")
    print("=" * 50)
