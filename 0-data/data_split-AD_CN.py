import os
import shutil

def copy_set_files(source_dir, destination_dir):
    """
    Copy .set files from the source directory to the destination directory.
    从源目录复制 .set 文件到目标目录。
    """
    # Ensure the destination directory exists; create it if it doesn't.
    # 确保目标目录存在，如果不存在则创建。
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Iterate through all files and subfolders in the source directory.
    # 遍历源目录中的所有文件和子文件夹。
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Check if the file extension is .set.
            # 检查文件扩展名是否为 .set。
            if file.endswith('.set'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(destination_dir, file)

                # Copy the file to the destination directory.
                # 复制文件到目标目录。
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {source_file} to {destination_file}")

    print("All .set files have been copied.")
    # 所有 .set 文件都已复制完成。

def read_file_list(file_path):
    """
    Read a list of filenames from a text file.
    从文本文件中读取文件名列表。
    """
    with open(file_path, 'r') as f:
        file_names = [line.strip() for line in f]
    return file_names

def move_files_by_list(source_dir, target_dir, file_list):
    """
    Move files from the source directory to the target directory based on a list of filenames.
    根据文件名列表将文件从源目录移动到目标目录。
    """
    # Ensure the target directory exists; create it if it doesn't.
    # 确保目标目录存在，如果不存在则创建。
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Move the specified files to the target directory.
    # 移动指定的文件到目标目录。
    for file_name in file_list:
        source_file_path = os.path.join(source_dir, file_name)
        target_file_path = os.path.join(target_dir, file_name)
        if os.path.exists(source_file_path):
            shutil.move(source_file_path, target_file_path)
            print(f"Moved: {source_file_path} to {target_file_path}")
        else:
            print(f"File {source_file_path} does not exist in the source directory.")
            # 文件 {source_file_path} 不存在于源目录中。

def main():
    # Step 1: Copy .set files to the data directory.
    # 第一步：复制 .set 文件到 data 目录。
    source_directory = 'derivatives'  # Replace with your source directory path
    # 替换为你的源文件夹路径。
    destination_directory = 'data'  # Replace with your destination directory path
    # 替换为你的目标文件夹路径。
    copy_set_files(source_directory, destination_directory)

    # Step 2: Move files to the corresponding directory based on file lists.
    # 第二步：根据文件列表将文件移动到相应的目录。
    list_directory = '0-data/AD_CN'  # Define the directory where filename lists are stored
    # 定义存储文件名列表的文本文件夹。
    categories = ['test', 'train']
    subfolders = ['CN', 'AD']
    data_directory = 'data'  # Path to the data folder
    # data 文件夹路径。

    for category in categories:
        for subfolder in subfolders:
            list_file = os.path.join(list_directory, f'{category}_{subfolder}_file_list.txt')
            target_directory = os.path.join('psd_AD_CN', category, subfolder)

            if os.path.exists(list_file):
                file_list = read_file_list(list_file)
                move_files_by_list(data_directory, target_directory, file_list)
            else:
                print(f"File list {list_file} does not exist.")
                # 文件列表 {list_file} 不存在。

if __name__ == '__main__':
    main()
