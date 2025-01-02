import os
import numpy as np

def process_files(input_folders, output_folders):
    """
    处理输入文件夹中的 .npy 文件并将其保存到输出文件夹。
    
    :param input_folders: 输入文件夹路径列表
    :param output_folders: 输出文件夹路径列表
    """
    # 确保输出文件夹存在
    for output_folder_path in output_folders:
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)

    # 遍历每个输入文件夹
    for input_folder_path, output_folder_path in zip(input_folders, output_folders):
        # 获取输入文件夹中的所有 .npy 文件
        npy_files = [f for f in os.listdir(input_folder_path) if f.endswith('.npy')]

        for npy_file in npy_files:
            # 完整文件路径
            file_path = os.path.join(input_folder_path, npy_file)
            
            # 加载数据
            data = np.load(file_path)
            
            # 确保数据的第二个维度是 95
            if data.shape[1] != 95:
                print(f"文件 {npy_file} 不是期望的形状 {data.shape}")
                continue
            
            # 保存每个通道的数据到单独的 .npy 文件中
            base_filename = os.path.splitext(npy_file)[0]
            for i in range(data.shape[0]):
                output_filename = f"{base_filename}_channel_{i}.npy"
                output_path = os.path.join(output_folder_path, output_filename)
                np.save(output_path, data[i])
                print(f"保存 {output_path} 成功!")

    print("所有文件处理完成!")

def main():
    # 选择路径列表
    set_1 = {
        "input": ['psd/test/AD', 'psd/test/CN', 'psd/train/AD', 'psd/train/CN'],
        "output": ['psd_data_cut/test/AD', 'psd_data_cut/test/CN', 'psd_data_cut/train/AD', 'psd_data_cut/train/CN']
    }

    set_2 = {
        "input": ['psd/test/FTD', 'psd/test/CN', 'psd/train/FTD', 'psd/train/CN'],
        "output": ['psd_data_cut/test/FTD', 'psd_data_cut/test/CN', 'psd_data_cut/train/FTD', 'psd_data_cut/train/CN']
    }

    # 在这里选择你要使用的路径集合
    selected_set = set_2

    process_files(selected_set["input"], selected_set["output"])

if __name__ == '__main__':
    main()
