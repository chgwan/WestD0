import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

def plot_all_scopes(mat_file_path):
   
    data = loadmat(mat_file_path)
    scope_keys = [key for key in data.keys() if key.endswith('_scope')]
    # with open("a.txt", 'w') as f:
    #     for scope_key in scope_keys:
    #         f.write(f'"{scope_key}", ')
    print(scope_keys)
    for scope_name in scope_keys:
        # Each scope is expected to be a struct at data[scope_name][0,0]
        scope_struct = data[scope_name][0,0]
        time_field = scope_struct['time'].squeeze()
        
        # Extract signals and values
        signals_field = scope_struct['signals']
        signal_values = signals_field['values'][0,0]  
        
       
        num_signals = signal_values.shape[1]
        # for i in range(num_signals):
        #     plt.figure()
        #     plt.plot(time_field, signal_values[:, i])
        #     plt.title(f'{scope_name} Signal {i}')
        #     plt.xlabel('Time')
        #     plt.ylabel('Signal Amplitude')
        #     # plt.show()
        #     plt.savefig(f"{scope_name}_{i}.png")


plot_all_scopes('/donnees/NTU/NTU/DCS_archive_57604.mat')