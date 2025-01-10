import numpy as np
from scipy.signal import welch
import os

def calculate_psd(signal, fs, nperseg):
    """
    使用Welch方法计算功率谱密度（PSD）。

    参数:
    - signal: 输入信号（二维数组：通道数 x 样本数）。
    - fs: 采样频率（Hz）。
    - nperseg: 每段的长度（样本数）。

    返回:
    - f: 频率数组。
    - psd: 功率谱密度（PSD），形状为（通道数 x 频率数）。
    """
    psd_list = []
    for ch_signal in signal:
        f, psd_ch = welch(ch_signal, fs, nperseg=nperseg)
        psd_list.append(psd_ch)
    return f, np.array(psd_list)

def calculate_rbp(f, psd, freq_bands):
    """
    计算相对波动指数（RBP）。

    参数:
    - f: 频率数组。
    - psd: 功率谱密度（PSD），形状为（通道数 x 频率数）。
    - freq_bands: 频段列表，每个频段是一个元组（开始频率，结束频率）。

    返回:
    - rbp: 相对波动指数（RBP），形状为（通道数 x 频段数）。
    """
    total_psd = np.sum(psd, axis=1, keepdims=True)
    rbp = []
    for (f_start, f_end) in freq_bands:
        band_power = np.sum(psd[:, (f >= f_start) & (f <= f_end)], axis=1, keepdims=True)
        rbp.append(band_power / total_psd)
    return np.hstack(rbp)

# 示例频段（可以根据实际需求调整）
freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 45)]

# 处理一个文件夹中的所有 .npy 文件
input_folder = 'data_npy_cut/test/AD'  # 输入文件夹路径
output_folder = 'PSD-RBP/test/AD'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理文件夹中的所有 .npy 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename.split('.')[0] + '_rbp.npy')
        
        # 加载输入数据
        signal = np.load(input_file_path)

        # 检查输入信号的大小是否为 [19, 2500]
        assert signal.shape == (19, 2500), f"Expected input signal shape to be (19, 2500), but got {signal.shape}"

        # 设置采样频率和每段的长度
        fs = 500  # 采样频率（Hz）
        nperseg = 128  # 每段的长度（样本数）

        # 计算PSD
        f, psd = calculate_psd(signal, fs, nperseg)

        # 计算RBP
        rbp = calculate_rbp(f, psd, freq_bands)

        # 将RBP保存到新的 .npy 文件
        np.save(output_file_path, rbp)
        print(f'Saved {output_file_path}')
