from .darpa_handler import DARPAHandler
from .atlas_handler import ATLASHandler

__all__ = ["DARPAHandler", "ATLASHandler"]

handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "atlas": ATLASHandler}



def get_handler(name, train, PATH_MAP, **kwargs):
    cls = handler_map.get(name.lower())
    base_path = PATH_MAP.get(name)
    if base_path is None:
        raise ValueError(f"未配置数据路径: {name}")
    if cls is None:
        raise ValueError(f"未知数据集: {name}")
    
    # 如果是ATLAS数据集且启用时间分割，传递额外参数
    if name.lower() == "atlas":
        return cls(base_path, train)
    else:
        return cls(base_path, train)
