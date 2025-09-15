from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "atlas": ATLASHandler}

path_map = {
    #"theia": "D:/data_files/theia",
    "theia": "/mnt/bigdata/aptdata/data_files/theia",
    "cadets": "/home/nsas2020/fuzz/Flash-IDS/data_files/cadets",
    "clearscope": "/home/nsas2020/fuzz/Flash-IDS/data_files/clearscope",
    "trace": "/home/nsas2020/fuzz/Flash-IDS/data_files/trace",
   "atlas": "/mnt/bigdata/aptdata/atlas_data",
    # "atlas": "D:/atlas_data_short",
}

def get_handler(name, train, use_time_split=False, **kwargs):
    cls = handler_map.get(name.lower())
    base_path = path_map.get(name)
    if base_path is None:
        raise ValueError(f"未配置数据路径: {name}")
    if cls is None:
        raise ValueError(f"未知数据集: {name}")
    
    # 如果是ATLAS数据集且启用时间分割，传递额外参数
    if name.lower() == "atlas" and use_time_split:
        return cls(base_path, train, use_time_split=True, **kwargs)
    else:
        return cls(base_path, train)
