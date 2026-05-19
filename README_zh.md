# FPSD-CNN: 基于扁平化功率谱密度和CNN的阿尔茨海默病EEG分类增强方法

本仓库包含 FPSD-CNN 方法的官方实现代码，用于使用 EEG 信号进行阿尔茨海默病分类，相关论文如下：

> Li, Z., Chen, X., Chen, L. et al. Enhanced EEG Classification for Alzheimer's Disease Using Flattened Power Spectral Density and CNN. Neural Process Lett (2026). https://doi.org/10.1007/s11063-026-11850-5

[English](README.md) | [中文](README_zh.md)

## 摘要

阿尔茨海默病（AD）是一种进行性痴呆症。脑电图（EEG）为AD诊断和与其他痴呆症的区分提供了有希望的途径，但EEG数据的复杂性和噪声带来了挑战。本研究提出FPSD-CNN，将频率功率谱密度（PSD）分析与深度卷积神经网络（CNN）相结合，以提高EEG信号的AD分类准确率。

我们在公开EEG数据集上进行二分类：
- **AD患者 vs. 认知正常（CN）受试者**
- **CN受试者 vs. 额颞叶痴呆（FTD）患者**

数据集包含88名受试者，其中36名AD患者（平均年龄=66.4，标准差=7.9），23名FTD患者（平均年龄=63.6，标准差=8.2），以及29名CN受试者（平均年龄=67.9，标准差=5.4）。

**主要结果：**
- AD vs. CN 分类准确率：**88.32%**
- CN vs. FTD 分类准确率：**82.22%**

## 仓库结构

```
FPSD-CNN-main/
├── 0-data/                    # 数据目录
│   ├── AD_CN/                 # AD vs CN 分类数据
│   │   ├── test_AD_file_list.txt
│   │   ├── test_CN_file_list.txt
│   │   ├── train_AD_file_list.txt
│   │   └── train_CN_file_list.txt
│   ├── CN_FTD/                # CN vs FTD 分类数据
│   │   ├── test_CN_file_list.txt
│   │   ├── test_FTD_file_list.txt
│   │   ├── train_CN_file_list.txt
│   │   └── train_FTD_file_list.txt
│   ├── README.md
│   ├── data_split-AD_CN.py
│   └── data_split-CN_FDT.py
├── 1-data_processing/         # 数据处理脚本
│   ├── 1-psd.py              # PSD特征提取
│   ├── 2-fpsd-processing.py  # FPSD处理
│   └── 3-Create-list.py      # 创建数据列表
├── 2-model/                   # 模型实现
│   ├── fpsd-cnn.py           # 主FPSD-CNN模型
│   └── fpsd-cnn-test.py      # 模型测试
└── ablation_study_code/       # 消融实验代码
    ├── 2D-CNN.py
    ├── 2D-CNN_No_AF.py
    ├── psd-cnn-ELU.py
    ├── psd-cnn-RELU.py
    ├── psd-cnn-SELU.py
    ├── psd-cnn-Softplus.py
    ├── psd-cnn_Leaky-Relu.py
    └── psd-cnn_No_AF.py
```

## 方法

### FPSD特征提取
1. **PSD计算**：使用频域分析从EEG信号中提取功率谱密度特征
2. **扁平化**：将2D PSD特征转换为1D扁平化PSD（FPSD）特征
3. **CNN分类**：将FPSD特征输入CNN进行自动特征提取和分类

### 模型架构
FPSD-CNN模型集成了：
- 频域PSD分析，用于捕获EEG信号特征
- 深度CNN，用于自动特征学习和分类
- 在原始EEG和去伪迹数据上评估噪声鲁棒性

## 使用方法

### 数据准备
1. 将EEG数据组织在`0-data/`目录中
2. 运行数据划分脚本：
   ```bash
   python 0-data/data_split-AD_CN.py
   python 0-data/data_split-CN_FDT.py
   ```

### 特征提取
```bash
cd 1-data_processing/
python 1-psd.py
python 2-fpsd-processing.py
python 3-Create-list.py
```

### 模型训练
```bash
cd 2-model/
python fpsd-cnn.py
```

### 模型测试
```bash
cd 2-model/
python fpsd-cnn-test.py
```

## 机构

1. 成都锦城学院，四川成都
2. 西安交通大学，陕西西安
3. 北京大学华东生物技术研究所，江苏启东

## 引用

如果您使用此代码或发现我们的工作有帮助，请引用我们的论文：

```bibtex
@article{li2026enhanced,
  title={Enhanced EEG Classification for Alzheimer's Disease Using Flattened Power Spectral Density and CNN},
  author={Li, Zhengji and Chen, Xing and Chen, Li and Lan, Yiheng and Luo, Junqi and Chen, Chuxi and Chen, Shiqing and Wei, Xin and Yang, Dan},
  journal={Neural Processing Letters},
  year={2026},
  doi={10.1007/s11063-026-11850-5}
}
```

## 许可证

本工作采用知识共享署名4.0国际许可证。