# -*- coding: utf-8 -*-
import os
import pathlib
from typing import Dict, Optional

from .utils import load_yaml_config


def _expand_path(s: str) -> pathlib.Path:
    return pathlib.Path(os.path.expandvars(s))


class ProjConfig:
    eps: float = 1e-7
    base_dir: pathlib.Path = pathlib.Path(__file__).resolve().parent.parent
    base_config_f: pathlib.Path = base_dir / 'configs' / 'base.yml'
    base_config: Dict = load_yaml_config(base_config_f)
    proj_db_dir: pathlib.Path = base_dir / 'ProjDB'

    # All data directories derived from a single database_dir
    database_dir: pathlib.Path = _expand_path(base_config['data']['database_dir'])
    data_dir: pathlib.Path = database_dir / 'WestData'
    pcs_dir: pathlib.Path = database_dir / 'PCS'
    imash5_dir: pathlib.Path = database_dir / 'IMASH5'
    error_dir: pathlib.Path = database_dir / 'ErrorShots'
    stat_dir: pathlib.Path = proj_db_dir / 'Stat'
    figs_dir: pathlib.Path = proj_db_dir / 'ConFigs'


_proj_config: Optional[ProjConfig] = None


def get_proj_config() -> ProjConfig:
    global _proj_config
    if _proj_config is None:
        _proj_config = ProjConfig()
    return _proj_config
