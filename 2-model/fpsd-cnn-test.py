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
        self.inplace = inplace  #将 inplace 参数保存为实例变量
        self.relu = nn.ReLU(inplace=inplace)   #创建一个ReLU激活函数实例，并根据 inplace 参数决定是否进行就地操作。
        self.scale = nn.Parameter(scale_value * torch.ones(1),  
            requires_grad=scale_learnable)#初始化一个可学习的缩放参数，其初始值为 scale_value，并根据 scale_learnable 决定它是否可学习。
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)#初始化一个可学习的偏置参数，其初始值为 bias_value，并根据 bias_learnable 决定它是否可学习。
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias
    # StarReLU: s * relu(x) ** 2 + b激活公式

# 自定义数据集类，从CSV文件中加载数据
class CsvDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):#idx是索引
        file_path = self.data.iloc[idx, 0]#第idx行0列的数据
        label = self.data.iloc[idx, 1]#第idx行1列的标签文件
        data = np.load(file_path)
        #使用NumPy库的load函数加载由file_path指定的文件。通常这个文件是一个.npy或.npz格式的NumPy数组。
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        # 这一行将加载的数据和标签转换为PyTorch张量（tensor）。data被转换为浮点数类型（torch.float32），而label被转换为长整型（torch.long）。然后，这两个张量作为一个元组返回。


# 定义简单的卷积神经网络模型
class CNN(nn.Module):
    def __init__(self,num_classes=2):
        super(CNN, self).__init__()
        # self.conv1 = nn.Conv1d(in_channels=输入通道, out_channels=输出通道, kernel_size=核卷积大小)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)#第一个卷积层
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3)#第二个卷积层
        # self.pool = nn.MaxPool1d(kernel_size=池化窗口大小, stride=步长)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 22, 128)  # 根据实际情况调整输入大小   定义了第一个全连接层，输入特征数为32 * 22，输出特征数为128。这里的输入特征数需要根据实际情况进行调整。
        self.fc2 = nn.Linear(128, num_classes)  # 假设有2个类别         定义了第二个全连接层，输入特征数为128，输出特征数为num_classes。
        self.futool = StarReLU()

    def forward(self, x):
        x = self.conv1(x)#输入张量 第一个卷积层处理
        # print('1',x.shape)
        x = self.futool(x)#激活函数
        x = self.pool(x)#池化处理
        # print('2',x.shape)
        x = self.conv2(x)#输入张量  第二个卷积层处理
        # print('3',x.shape)
        x = self.futool(x)#激活函数
        x = self.pool(x)#池化处理
        # print('4',x.shape)
        x = x.view(-1, 32 * 22)#将池化层的结果展平成一个二维张量，其中-1表示自动计算该维度的大小，32 * 22是展平后的特征维度。
        # print('4-1',x.shape)
        x = self.fc1(x) #线性处理
        # print('5',x.shape)
        x = self.futool(x) #激活函数，非线性处理
        # print('6',x.shape)
        x = self.fc2(x)#全连接为目标输出特征
        # print('7',x.shape)
        return x



# 设置中文显示
rcParams['font.family'] = 'SimHei'#图表中的标题和轴标签都使用了中文黑体字体。这样，即使是在英文环境中，也能正确显示中文。

# labels = ['AD', 'CN']
labels = ['FTD','CN']#定义了一个标签列表
softmax = nn.Softmax(dim=1)#每个样本的输出将被转换为一个概率分布，其中每个元素代表该样本属于相应类别的概率。

def val(batch_size=1):
    # 数据集和数据加载器
    val_dataset =  CsvDataset(csv_file='test_data.csv')#CsvDataset类会包含从CSV文件读取数据并转换为模型可用的格式的方法。
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    #DaraLoader(val_dataset,batch_size=参数指定了每个批次的大小,shuffle=True:开始时打乱顺序，drop_last=True：如果最后一个数据不足一个的时候丢弃)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #这行代码根据是否有可用的CUDA设备（即NVIDIA GPU）来选择使用CPU还是GPU进行计算。

    model = CNN(num_classes=2).to(device)
    # 这行代码创建了一个CNN模型实例，该模型被设计为处理2个类别的分类任务，并将其移动到之前选择的设备（CPU或GPU）上。
    model.load_state_dict(torch.load("best_model.pth"))
    # 从best_model.pth文件中加载模型的权重和状态，并将其应用到当前模型实例上。

    arr_y = []
    arr_y_pred = []
    #这两个列表用于存储真实标签和预测标签
    for val_x, val_y in val_data_loader:
        #这行代码开始一个循环，每次迭代从数据加载器中获取一个批次的数据。
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        #将这俩个数据移到这个设备上（cpu or gpu）
        val_x = val_x.unsqueeze(1)
        #数据预处理，增加一个维度，以便模型正确处理这个数据
        val_y_pred = model(val_x)#使用模型对当前批次的数据进行预测。
        arr_y.extend(val_y.cpu().numpy())#将代码的真实标签移到cpu，并且处理为numpy数组，并且添加到arr_y的列表里面
        pred_result = softmax(val_y_pred).max(dim=1)[1]
        #对模型的预测结果应用softmax函数，然后选择概率最高的类别作为预测结果。
        arr_y_pred.extend(pred_result.cpu().numpy())
        #将预测结果从GPU移动到CPU，并将其转换为NumPy数组，然后将其添加到arr_y_pred列表中。

    # 计算特异度
    cm = confusion_matrix(arr_y, arr_y_pred) #求得真实标签和预测标签之前的混淆矩阵
    # 计算特异度（specificity）
    TN = cm[1, 1]  # True Negative   实际为负类且被模型预测为负类的样本数量。
    FP = cm[1, 0]  # False Positive   实际为正类但被模型预测为负类的样本数量。
    specificity = TN / (TN + FP)
    #计算特异度，即TN除以TN和FP之和。
    print(f'Specificity: {specificity}')

    # 计算准确率（Accuracy）
    accuracy = np.diag(cm).sum() / cm.sum() 
    print(f'Accuracy: {accuracy}')

    # 计算精确率（Precision）
    precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) #
    print(f'Precision: {precision}')

    # 计算召回率（Recall）
    recall = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    print(f'Recall: {recall}')

    # 计算F1分数（F1 score）
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}, Specificity: {specificity:.5f}")


    plt.imshow(cm, cmap="Blues") #用蓝色来表示这个混淆矩阵的数值大小
    plt.xticks(range(len(labels)), labels=labels)
    plt.yticks(range(len(labels)), labels=labels)

    
    plt.xlabel("Prediction", fontsize=14)
    plt.ylabel("Reference", fontsize=14)
    plt.title('Data Without Artifact Processing CN-FTD')
    thresh = cm.mean()
    for i in range(len(labels)):
        for j in range(len(labels)):
            info = cm[j, i]
            prob = info / np.sum(cm[j]) #即该类别在所有预测为该类别的样本中的比例。
            plt.text(i, j, f"{info}\n({prob*100:.2f}%)", color="white" if info > thresh else "black", ha='center', va='center')#是要显示的文本，包括单元格的值和概率
    
    plt.savefig("confusion_matrix.jpg")
    # plt.savefig 是一个函数，用于保存图像。
   # "confusion_matrix.jpg" 是保存的文件名。
    plt.show()

if __name__ == "__main__":
    val()