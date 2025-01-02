import os
import csv
import numpy as np

train_path = "train_data.csv"
val_path = "test_data.csv"

def create_data_text(path):
    """建立数据data列表,划分数据集"""
    f_train = open(train_path, "w", newline='') 
    f_val = open(val_path, "w", newline='')
    
    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    
    # 遍历 'train' 文件夹
    train_dir = os.path.join(path, 'train')
    for cls, dirname in enumerate(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, dirname)
        if os.path.isdir(class_path):
            flist = os.listdir(class_path)
            np.random.shuffle(flist)
            fnum = len(flist)
            for i, filename in enumerate(flist):
                train_writer.writerow([os.path.join(class_path, filename), str(cls)])
    
    # 遍历 'test' 文件夹
    test_dir = os.path.join(path, 'test')
    for cls, dirname in enumerate(os.listdir(test_dir)):
        class_path = os.path.join(test_dir, dirname)
        if os.path.isdir(class_path):
            flist = os.listdir(class_path)
            np.random.shuffle(flist)
            fnum = len(flist)
            for i, filename in enumerate(flist):
                val_writer.writerow([os.path.join(class_path, filename), str(cls)])

    f_train.close()
    f_val.close()

if __name__ == "__main__":
    create_data_text("psd_data_cut")
    print("列表创建完成")
