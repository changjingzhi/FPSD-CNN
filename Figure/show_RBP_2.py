import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
ad_npy_file_path = 'PSD-RBP/test/AD/sub-001_task-eyesclosed_eeg_cut_0_rbp.npy'  # 替换为实际的AD .npy 文件路径
cn_npy_file_path = 'RBP-npy/CN/sub-037_task-eyesclosed_eeg_cut_0_rbp.npy'  # 替换为实际的CN .npy 文件路径
fdt_npy_file_path = 'RBP-npy/FDT/sub-066_task-eyesclosed_eeg_cut_0_rbp.npy'  # 替换为实际的FDT .npy 文件路径

# 读取数据
ad_data = np.load(ad_npy_file_path)
cn_data = np.load(cn_npy_file_path)
fdt_data = np.load(fdt_npy_file_path)

# 检查数据形状
assert ad_data.shape == (19, 5), f"Expected AD data shape to be (19, 5), but got {ad_data.shape}"
assert cn_data.shape == (19, 5), f"Expected CN data shape to be (19, 5), but got {cn_data.shape}"
assert fdt_data.shape == (19, 5), f"Expected FDT data shape to be (19, 5), but got {fdt_data.shape}"

# 通道和频段标签
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
freq_bands = ['0.5-4 Hz', '4-8 Hz', '8-13 Hz', '13-25 Hz', '25-45 Hz']

# 创建图形和子图
fig, axs = plt.subplots(1, 3, figsize=(8, 8), sharey=True)

# 绘制AD数据热图
axs[0].matshow(ad_data, cmap='viridis')
axs[0].set_title('AD Power Spectral Density')
axs[0].set_xticks(np.arange(len(freq_bands)))
axs[0].set_xticklabels(freq_bands, rotation=45, ha='left')
axs[0].set_yticks(np.arange(len(channels)))
axs[0].set_yticklabels(channels)

# 绘制CN数据热图
axs[1].matshow(cn_data, cmap='viridis')
axs[1].set_title('CN Power Spectral Density')
axs[1].set_xticks(np.arange(len(freq_bands)))
axs[1].set_xticklabels(freq_bands, rotation=45, ha='left')
axs[1].set_yticks(np.arange(len(channels)))
axs[1].set_yticklabels(channels)

# 绘制FDT数据热图
axs[2].matshow(fdt_data, cmap='viridis')
axs[2].set_title('FTD Power Spectral Density')
axs[2].set_xticks(np.arange(len(freq_bands)))
axs[2].set_xticklabels(freq_bands, rotation=45, ha='left')
axs[2].set_yticks(np.arange(len(channels)))
axs[2].set_yticklabels(channels)

# 设置总体标签
fig.suptitle(' Power Spectral Density (PSD) Comparison', fontsize=16)
fig.text(0.5, 0.04, 'Data Without Artifact Processing', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Channels', va='center', rotation='vertical', fontsize=14)

# 显示图像
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # 调整布局以防止标签重叠
plt.savefig('psd_comparison_high_quality.png', dpi=1024, bbox_inches='tight')
plt.show()

