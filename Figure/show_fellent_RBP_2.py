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

# 将数据按列展平为 (1, 95)
ad_data_reshaped = ad_data.T.reshape(1, -1)  # 变形为 (1, 95)
cn_data_reshaped = cn_data.T.reshape(1, -1)  # 变形为 (1, 95)
fdt_data_reshaped = fdt_data.T.reshape(1, -1)  # 变形为 (1, 95)

# 创建图形和子图
fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 绘制AD数据条形图
axs[0].bar(np.arange(ad_data_reshaped.shape[1]), ad_data_reshaped.flatten(), color='blue')
axs[0].set_title('AD Flatten Power Spectral Density')
axs[0].set_ylabel('Value')

# 绘制CN数据条形图
axs[1].bar(np.arange(cn_data_reshaped.shape[1]), cn_data_reshaped.flatten(), color='green')
axs[1].set_title('CN Flatten Power Spectral Density')
axs[1].set_ylabel('Value')

# 绘制FDT数据条形图
axs[2].bar(np.arange(fdt_data_reshaped.shape[1]), fdt_data_reshaped.flatten(), color='orange')
axs[2].set_title('FTD Flatten Power Spectral Density')
axs[2].set_xlabel('Data Points')
axs[2].set_ylabel('Value')

# 设置总体标签
fig.suptitle('Flatten Power Spectral Density (PSD) Comparison', fontsize=16)
fig.text(0.5, 0.04, 'Data Without Artifact Processing', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Channels', va='center', rotation='vertical', fontsize=14)
# 调整布局以防止标签重叠
plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
plt.savefig('fpsd_comparison_high_quality.png', dpi=1024, bbox_inches='tight')
plt.show()