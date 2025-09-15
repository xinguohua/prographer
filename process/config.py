"""
时间分割配置文件
用于管理基于时间戳的数据分割参数
"""

# =====================================================
# 时间分割参数配置
# =====================================================

# 恶意事件时间窗口（分钟）
# 恶意集 = 恶意事件前后 ± MALICIOUS_WINDOW_MINUTES 分钟的数据
MALICIOUS_WINDOW_MINUTES = 10

# 测试事件时间窗口（分钟）  
# 测试集 = 恶意事件前后 ± TEST_WINDOW_MINUTES 分钟的数据
TEST_WINDOW_MINUTES = 20

# 训练集 = 不在测试窗口内的所有其他数据

# =====================================================
# 模型训练/测试参数配置
# =====================================================

# 序列长度参数
SEQUENCE_LENGTH_L = 12          # 原始论文中的序列长度
MIN_SEQUENCE_LENGTH = 3         # 最小序列长度（当快照不足时）
SEQUENCE_ADAPT_RATIO = 0.5      # 动态调整比例（快照数量的一半）

# 检测阈值
DETECTION_THRESHOLD = 0.01      # 异常检测阈值

# 快照生成参数
SNAPSHOT_SIZE = 500             # 快照大小（节点数）
FORGETTING_RATE = 0.2           # 遗忘率

# =====================================================
# 文件路径配置
# =====================================================

# 模型保存路径
DETECTOR_MODEL_PATH = "D:/prographer/process/prographer_detector.pth"
ENCODER_MODEL_PATH = "D:/prographer/process/prographer_encoder.pth"

# 数据集路径（在 __init__.py 中配置）
# 分割后的数据集文件命名格式：
# - {graph_name}_trainlogs.csv    (训练集)
# - {graph_name}_testlogs.csv     (测试集)  
# - {graph_name}_maliciouslogs.csv (恶意集)

# =====================================================
# 调试和日志配置
# =====================================================

# 是否启用详细调试信息
ENABLE_DEBUG_INFO = True

# 是否保存数据分割质量报告
SAVE_SPLIT_QUALITY_REPORT = True

# 是否显示时间窗口可视化信息
SHOW_TIME_WINDOW_INFO = True

def get_time_split_config():
    """获取时间分割配置字典"""
    return {
        'malicious_window_minutes': MALICIOUS_WINDOW_MINUTES,
        'test_window_minutes': TEST_WINDOW_MINUTES,
        'sequence_length': SEQUENCE_LENGTH_L,
        'min_sequence_length': MIN_SEQUENCE_LENGTH,
        'sequence_adapt_ratio': SEQUENCE_ADAPT_RATIO,
        'detection_threshold': DETECTION_THRESHOLD,
        'snapshot_size': SNAPSHOT_SIZE,
        'forgetting_rate': FORGETTING_RATE,
        'enable_debug': ENABLE_DEBUG_INFO
    }

def print_config_summary():
    """打印配置摘要"""
    print("\n" + "="*50)
    print(" 时间分割配置摘要")
    print("="*50)
    print(f" 恶意时间窗口: ±{MALICIOUS_WINDOW_MINUTES} 分钟")
    print(f" 测试时间窗口: ±{TEST_WINDOW_MINUTES} 分钟")
    print(f" 序列长度: {SEQUENCE_LENGTH_L} (最小: {MIN_SEQUENCE_LENGTH})")
    print(f" 检测阈值: {DETECTION_THRESHOLD}")
    print(f" 快照大小: {SNAPSHOT_SIZE} 节点")
    print(f" 遗忘率: {FORGETTING_RATE}")
    print("="*50)
