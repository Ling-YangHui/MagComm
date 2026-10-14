import pickle
import os
import subprocess
import pprint
import numpy as np
from scipy.integrate import trapezoid
from pa.lpf import *
from pa.resample import *
import matplotlib.pyplot as plt

ds_path = "C:/Users/YangHui/Documents/ADS/MagnetWirelessCommunication/data/basic_lc_simulation.ds"

output_file_name = 'basic_lc_simulation'
script_path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(script_path, "..", "..", "out", f'{output_file_name}.pkl')

if not os.path.exists(out_path):
    # 在对应目录下执行bat脚本提取数据
    run_path = os.path.join(script_path, "..", "..")
    subprocess.run([os.path.join(run_path, "run_export_ds.bat"), ds_path], cwd=run_path, shell=True)

data = pickle.load(open(out_path, 'rb'))
pprint.pprint(data['Tran1.TRAN'].keys())

time = data['Tran1.TRAN']['time']
current = data['Tran1.TRAN']['Vrx']
np.save("test_signal.npy", np.array(current))
frequency = 4e3

current, time = resample_signal(current, time, 32e3)
nco = np.exp(-1j * 2 * np.pi * frequency * time)
current_demodulated = fir_lowpass_filter(current * nco, 100, 32e3, 101)

plt.plot(time * 1e3, np.angle(current_demodulated), label='Demodulated Current (Real)', color='b')
plt.plot(time * 1e3, np.abs(current_demodulated), label='Demodulated Current (Magnitude)', color='r')
plt.grid(True)
plt.legend()
plt.xlabel('Time (ms)')
plt.show()