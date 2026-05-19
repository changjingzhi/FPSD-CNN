# FPSD-CNN: Enhanced EEG Classification for Alzheimer's Disease Using Flattened Power Spectral Density and CNN

This repository contains the official implementation of the FPSD-CNN method for Alzheimer's disease classification using EEG signals, as described in the paper:

> Li, Z., Chen, X., Chen, L. et al. Enhanced EEG Classification for Alzheimer's Disease Using Flattened Power Spectral Density and CNN. Neural Process Lett (2026). https://doi.org/10.1007/s11063-026-11850-5

[English](README.md) | [中文](README_zh.md)

## Abstract

Alzheimer's disease (AD) is a progressive form of dementia. Electroencephalography (EEG) offers a promising avenue for AD diagnosis and differentiation from other dementias, but EEG data complexity and noise have posed challenges. This study proposes FPSD-CNN, integrating Frequency Power Spectral Density (PSD) analysis with a deep convolutional neural network (CNN) to enhance AD classification accuracy from EEG signals.

We perform binary classification on a public EEG dataset:
- **AD vs. cognitively normal (CN) subjects**
- **CN vs. frontotemporal dementia (FTD) subjects**

The dataset includes 88 subjects, with 36 AD patients (mean age=66.4, sd=7.9), 23 FTD patients (mean age=63.6, sd=8.2), and 29 CN subjects (mean age=67.9, sd=5.4).

**Key Results:**
- AD vs. CN classification: **88.32% accuracy**
- CN vs. FTD classification: **82.22% accuracy**

## Repository Structure

```
FPSD-CNN-main/
├── 0-data/                    # Data directory
│   ├── AD_CN/                 # AD vs CN classification data
│   │   ├── test_AD_file_list.txt
│   │   ├── test_CN_file_list.txt
│   │   ├── train_AD_file_list.txt
│   │   └── train_CN_file_list.txt
│   ├── CN_FTD/                # CN vs FTD classification data
│   │   ├── test_CN_file_list.txt
│   │   ├── test_FTD_file_list.txt
│   │   ├── train_CN_file_list.txt
│   │   └── train_FTD_file_list.txt
│   ├── README.md
│   ├── data_split-AD_CN.py
│   └── data_split-CN_FDT.py
├── 1-data_processing/         # Data processing scripts
│   ├── 1-psd.py              # PSD feature extraction
│   ├── 2-fpsd-processing.py  # FPSD processing
│   └── 3-Create-list.py      # Create data lists
├── 2-model/                   # Model implementation
│   ├── fpsd-cnn.py           # Main FPSD-CNN model
│   └── fpsd-cnn-test.py      # Model testing
└── ablation_study_code/       # Ablation study implementations
    ├── 2D-CNN.py
    ├── 2D-CNN_No_AF.py
    ├── psd-cnn-ELU.py
    ├── psd-cnn-RELU.py
    ├── psd-cnn-SELU.py
    ├── psd-cnn-Softplus.py
    ├── psd-cnn_Leaky-Relu.py
    └── psd-cnn_No_AF.py
```

## Methodology

### FPSD Feature Extraction
1. **PSD Calculation**: Extract Power Spectral Density features from EEG signals using frequency-domain analysis
2. **Flattening**: Convert 2D PSD features into 1D Flattened PSD (FPSD) features
3. **CNN Classification**: Input FPSD features into a CNN for automated feature extraction and classification

### Model Architecture
The FPSD-CNN model integrates:
- Frequency-domain PSD analysis for capturing EEG signal characteristics
- Deep CNN for automated feature learning and classification
- Noise robustness evaluation on both raw EEG and artifact-removed data

## Usage

### Data Preparation
1. Organize EEG data in the `0-data/` directory
2. Run data splitting scripts:
   ```bash
   python 0-data/data_split-AD_CN.py
   python 0-data/data_split-CN_FDT.py
   ```

### Feature Extraction
```bash
cd 1-data_processing/
python 1-psd.py
python 2-fpsd-processing.py
python 3-Create-list.py
```

### Model Training
```bash
cd 2-model/
python fpsd-cnn.py
```

### Model Testing
```bash
cd 2-model/
python fpsd-cnn-test.py
```

## Authors

- Zhengji Li
- Xing Chen
- Li Chen
- Yiheng Lan
- Junqi Luo
- Chuxi Chen
- Shiqing Chen
- Xin Wei* (Corresponding author)
- Dan Yang

## Affiliations

1. Chengdu Jincheng College, Chengdu, Sichuan, China
2. Xi'an Jiaotong University, Xi'an, Shaanxi, China
3. East China Institute of Biotechnology, Peking University, Qidong, Jiangsu, China

## Citation

If you use this code or find our work helpful, please cite our paper:

```bibtex
@article{li2026enhanced,
  title={Enhanced EEG Classification for Alzheimer's Disease Using Flattened Power Spectral Density and CNN},
  author={Li, Zhengji and Chen, Xing and Chen, Li and Lan, Yiheng and Luo, Junqi and Chen, Chuxi and Chen, Shiqing and Wei, Xin and Yang, Dan},
  journal={Neural Processing Letters},
  year={2026},
  doi={10.1007/s11063-026-11850-5}
}
```

## License

This work is licensed under a Creative Commons Attribution 4.0 International License.