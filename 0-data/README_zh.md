# 数据目录说明

本目录包含数据获取和数据划分相关的文件，用于论文复现。

[English](README.md) | [中文](README_zh.md)

## 数据集下载

本研究使用的公开 EEG 数据集可从以下链接下载：

[OpenNeuro Dataset ds004504 (Version 1.0.2)](https://openneuro.org/datasets/ds004504/versions/1.0.2)

## 数据结构

下载完成后，数据集包含两种状态的数据：

| 数据类型 | 位置 | 说明 |
| :--- | :--- | :--- |
| 伪迹处理后 | `derivatives/` 文件夹内 | 已去除伪迹的干净 EEG 数据 |
| 原始数据 | `derivatives/` 文件夹外 | 未经过伪迹处理的原始 EEG 数据 |

## 数据划分文件

本目录包含以下数据划分配置文件：

### AD vs CN 分类
- `AD_CN/train_AD_file_list.txt` - AD 训练集文件列表
- `AD_CN/train_CN_file_list.txt` - CN 训练集文件列表
- `AD_CN/test_AD_file_list.txt` - AD 测试集文件列表
- `AD_CN/test_CN_file_list.txt` - CN 测试集文件列表

### CN vs FTD 分类
- `CN_FTD/train_CN_file_list.txt` - CN 训练集文件列表
- `CN_FTD/train_FTD_file_list.txt` - FTD 训练集文件列表
- `CN_FTD/test_CN_file_list.txt` - CN 测试集文件列表
- `CN_FTD/test_FTD_file_list.txt` - FTD 测试集文件列表

## 使用方法

### 方法一：使用伪迹处理后的数据

1. 将下载的 `derivatives` 文件夹复制到 `FPSD-CNN-main/` 目录下
2. 执行数据划分脚本：
   ```bash
   python data_split-AD_CN.py    # AD vs CN 分类
   # 或
   python data_split-CN_FTD.py   # CN vs FTD 分类
   ```

### 方法二：使用原始数据（未处理伪迹）

1. 创建一个新文件夹（例如命名为 `raw_data`）
2. 将 `derivatives` 文件夹外的所有文件复制到新文件夹中
3. 修改划分脚本中的源目录路径：
   ```python
   source_directory = 'raw_data'  # 将 'derivatives' 替换为你的文件夹名称
   ```
4. 执行数据划分脚本：
   ```bash
   python data_split-AD_CN.py
   # 或
   python data_split-CN_FTD.py
   ```

## 脚本说明

| 脚本 | 功能 |
| :--- | :--- |
| `data_split-AD_CN.py` | 划分 AD vs CN 分类的训练集和测试集 |
| `data_split-CN_FTD.py` | 划分 CN vs FTD 分类的训练集和测试集 |

## 数据集统计

| 组别 | 样本数 | 平均年龄 | 标准差 |
| :--- | :--- | :--- | :--- |
| AD (阿尔茨海默病) | 36 | 66.4 | 7.9 |
| CN (认知正常) | 29 | 67.9 | 5.4 |
| FTD (额颞叶痴呆) | 23 | 63.6 | 8.2 |
| **总计** | **88** | - | - |