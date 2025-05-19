import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm




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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 22, 128)  # 根据实际情况调整输入大小
        self.fc2 = nn.Linear(128, 2)  # 假设有2个类别
        self.futool = nn.Softplus()

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


import torch

import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, save_path='best_model.pth'):
    model.train()
    total_samples = len(train_loader.dataset)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for data, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(data.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': running_loss / total_samples, 'accuracy': correct_predictions / total_samples})

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validate the model
        model.eval()
        total_samples_val = len(val_loader.dataset)
        running_loss_val = 0.0
        correct_predictions_val = 0

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data.unsqueeze(1))
                loss_val = criterion(outputs, labels)
                running_loss_val += loss_val.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions_val += (predicted == labels).sum().item()

            epoch_loss_val = running_loss_val / total_samples_val
            epoch_accuracy_val = correct_predictions_val / total_samples_val

            print(f'Validation Loss: {epoch_loss_val:.4f}, Accuracy: {epoch_accuracy_val:.4f}')

            if epoch_accuracy_val > best_val_acc:
                best_val_acc = epoch_accuracy_val
                torch.save(model.state_dict(), save_path)
                print(f'Saved the model with the best validation accuracy: {best_val_acc:.4f}')

    print('Training finished')





# CSV训练集文件路径
train_csv_file_path = 'train_data.csv'
# CSV测试集文件路径
test_csv_file_path = 'test_data.csv'

# 创建训练集和测试集数据集实例
train_dataset = CsvDataset(train_csv_file_path)
test_dataset = CsvDataset(test_csv_file_path)

# 创建训练集和测试集数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,drop_last=True)

# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

# 训练模型
train(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)


