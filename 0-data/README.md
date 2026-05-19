# Data Directory

This directory contains data acquisition and data splitting files for paper reproducibility.


## Dataset Download

The public EEG dataset used in this study can be downloaded from:

[OpenNeuro Dataset ds004504 (Version 1.0.2)](https://openneuro.org/datasets/ds004504/versions/1.0.2)

## Data Structure

After downloading, the dataset contains two types of data:

| Data Type | Location | Description |
| :--- | :--- | :--- |
| Artifact-removed | Inside `derivatives/` folder | Clean EEG data with artifacts removed |
| Raw | Outside `derivatives/` folder | Raw EEG data without artifact processing |

## Data Split Files

This directory contains the following data split configuration files:

### AD vs CN Classification
- `AD_CN/train_AD_file_list.txt` - Training list for AD subjects
- `AD_CN/train_CN_file_list.txt` - Training list for CN subjects
- `AD_CN/test_AD_file_list.txt` - Testing list for AD subjects
- `AD_CN/test_CN_file_list.txt` - Testing list for CN subjects

### CN vs FTD Classification
- `CN_FTD/train_CN_file_list.txt` - Training list for CN subjects
- `CN_FTD/train_FTD_file_list.txt` - Training list for FTD subjects
- `CN_FTD/test_CN_file_list.txt` - Testing list for CN subjects
- `CN_FTD/test_FTD_file_list.txt` - Testing list for FTD subjects

## Usage

### Method 1: Using Artifact-removed Data

1. Copy the downloaded `derivatives` folder to `FPSD-CNN-main/` directory
2. Run the data splitting script:
   ```bash
   python data_split-AD_CN.py    # For AD vs CN classification
   # or
   python data_split-CN_FTD.py   # For CN vs FTD classification
   ```

### Method 2: Using Raw Data (No Artifact Processing)

1. Create a new folder (e.g., named `raw_data`)
2. Copy all files outside the `derivatives` folder into the new folder
3. Modify the source directory path in the splitting script:
   ```python
   source_directory = 'raw_data'  # Replace 'derivatives' with your folder name
   ```
4. Run the data splitting script:
   ```bash
   python data_split-AD_CN.py
   # or
   python data_split-CN_FTD.py
   ```

## Script Description

| Script | Function |
| :--- | :--- |
| `data_split-AD_CN.py` | Split training and testing sets for AD vs CN classification |
| `data_split-CN_FTD.py` | Split training and testing sets for CN vs FTD classification |

## Dataset Statistics

| Group | Sample Size | Mean Age | SD |
| :--- | :--- | :--- | :--- |
| AD (Alzheimer's Disease) | 36 | 66.4 | 7.9 |
| CN (Cognitively Normal) | 29 | 67.9 | 5.4 |
| FTD (Frontotemporal Dementia) | 23 | 63.6 | 8.2 |
| **Total** | **88** | - | - |