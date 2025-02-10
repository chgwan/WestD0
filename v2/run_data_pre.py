# -*- coding: utf-8 -*-
import pathlib
import os
from src.utils import merge_mat_h5
from private_modules.utilities import save_to_file, parse_args, MpRun
from tqdm import tqdm

def save_merge():
    def warp_merge(shot):
        h5_file = D0_signal_dir.joinpath(f'{shot}.h5')
        mat_file = Mat_dir.joinpath(f'DCS_archive_{shot}.mat')
        data_dict = merge_mat_h5(h5_file, mat_file)
        west_f = WestData_dir.joinpath(f'{shot}.h5')
        save_to_file(west_f, data_dict, True)

    D0_signal_dir = os.path.expandvars("$DATABASE_PATH/DataBase/WEST/D0_signals")
    D0_signal_dir = pathlib.Path(D0_signal_dir)
    Mat_dir = D0_signal_dir.parent.joinpath("PCS")
    WestData_dir = D0_signal_dir.parent.joinpath("WestData")

    shots = [int(shot.parts[-1][:-3]) for shot in D0_signal_dir.iterdir()]
    mp_run = MpRun(num_workers=10)
    mp_run.mp_no_return_run_func(warp_merge, shots)

if __name__ == "__main__":
    args = parse_args()
    run_code = args.run_code.strip()
    run_code = f"{run_code}"
    eval(run_code)    