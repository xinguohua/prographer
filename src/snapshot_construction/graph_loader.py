"""Paper §IV.A - Provenance graph loader dispatch.

Maps a supported paper-profile dataset name to the
right parser class and constructs it with the requested scene filter.
"""
from typing import Optional

from .darpa_e3_parser import DARPAHandler
from .darpa_e5_parser import DARPAHandler5
from .optc_parser import OptcHandler
from .atlas_parser import ATLASHandler


__all__ = ["DARPAHandler", "DARPAHandler5", "OptcHandler", "ATLASHandler",
           "handler_map", "get_handler"]


handler_map = {
    "theia": DARPAHandler,
    "cadets": DARPAHandler,
    "clearscope": DARPAHandler,
    "trace": DARPAHandler,
    "cadets5": DARPAHandler5,
    "theia5": DARPAHandler5,
    "trace5": DARPAHandler5,
    "clearscope5": DARPAHandler5,
    "optcday1": OptcHandler,
    "atlas": ATLASHandler,
}


def get_handler(name, train, PATH_MAP, scene_name: Optional[str] = None):
    """Return a provenance-graph handler for the requested dataset.

    Parameters
    ----------
    name : str
        One of the keys in :data:`handler_map` (case-insensitive).
    train : bool
        Whether the handler loads the training split.
    PATH_MAP : dict
        Dataset-name -> root path mapping, read from ``configs/athena.yaml``.
    scene_name : Optional[str]
        Scene filter (e.g. ``cadets314``). ``None`` loads every scene under
        ``name``.
    """
    lower_name = name.lower()
    cls = handler_map.get(lower_name)
    path_entry = PATH_MAP.get(lower_name)
    parser_options = {}
    if isinstance(path_entry, dict):
        base_path = path_entry.get("root")
        if lower_name == "atlas":
            parser_options["source_timezone"] = path_entry.get("source_timezone", "-05:00")
    else:
        base_path = path_entry

    if cls is None or base_path is None:
        raise ValueError(f"unknown dataset or missing path: {name}")

    return cls(
        base_path,
        train,
        scene_name=scene_name,
        dataset_name=lower_name,
        **parser_options,
    )
