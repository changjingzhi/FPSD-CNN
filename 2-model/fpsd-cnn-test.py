import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rcParams


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

# 自定义数据集类，从CSV文件中加载数据
class CsvDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 定义简单的卷积神经网络模型
class CNN(nn.Module):
    def __init__(self,num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 22, 128)  # 根据实际情况调整输入大小
        self.fc2 = nn.Linear(128, num_classes)  # 假设有2个类别
        self.futool = StarReLU()

    def forward(self, x):
        x = self.conv1(x)
        # print('1',x.shape)
        x = self.futool(x)
        x = self.pool(x)
        # print('2',x.shape)
        x = self.conv2(x)
        # print('3',x.shape)
        x = self.futool(x)
        x = self.pool(x)
        # print('4',x.shape)
        x = x.view(-1, 32 * 22)
        # print('4-1',x.shape)
        x = self.fc1(x)
        # print('5',x.shape)
        x = self.futool(x)
        # print('6',x.shape)
        x = self.fc2(x)
        # print('7',x.shape)
        return x



# 设置中文显示
rcParams['font.family'] = 'SimHei'

# labels = ['AD', 'CN']
labels = ['FTD','CN']
softmax = nn.Softmax(dim=1)

def val(batch_size=1):
    # 数据集和数据加载器
    val_dataset =  CsvDataset(csv_file='test_data.csv')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    arr_y = []
    arr_y_pred = []
    for val_x, val_y in val_data_loader:
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        val_x = val_x.unsqueeze(1)
        val_y_pred = model(val_x)
        arr_y.extend(val_y.cpu().numpy())
        pred_result = softmax(val_y_pred).max(dim=1)[1]
        arr_y_pred.extend(pred_result.cpu().numpy())

    # 计算特异度
    cm = confusion_matrix(arr_y, arr_y_pred)
    # 计算特异度（specificity）
    TN = cm[1, 1]  # True Negative
    FP = cm[1, 0]  # False Positive
    specificity = TN / (TN + FP)
    print(f'Specificity: {specificity}')

    # 计算准确率（Accuracy）
    accuracy = np.diag(cm).sum() / cm.sum()
    print(f'Accuracy: {accuracy}')

    # 计算精确率（Precision）
    precision = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    print(f'Precision: {precision}')

    # 计算召回率（Recall）
    recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print(f'Recall: {recall}')

    # 计算F1分数（F1 score）
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}, Specificity: {specificity:.5f}")


    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels=labels)
    plt.yticks(range(len(labels)), labels=labels)

    
    plt.xlabel("Prediction", fontsize=14)
    plt.ylabel("Reference", fontsize=14)
    plt.title('Data Without Artifact Processing CN-FTD')
    thresh = cm.mean()
    for i in range(len(labels)):
        for j in range(len(labels)):
            info = cm[j, i]
            prob = info / np.sum(cm[j])
            plt.text(i, j, f"{info}\n({prob*100:.2f}%)", color="white" if info > thresh else "black", ha='center', va='center')
    
    plt.savefig("confusion_matrix.jpg")
    plt.show()

if __name__ == "__main__":
    val()