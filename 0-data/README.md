这个文件家中的代码是数据获取和数据划分文件。为了方便论文的复现，将会提供这篇论文中的数据划分配置和随机划分数据集的选项。

公开数据集下载链接。[链接](https://openneuro.org/datasets/ds004504/versions/1.0.2)

下载完成后，derivatives文件夹中是已经处理过伪迹的数据集，而derivatives文件夹外的文件夹中是未经过伪迹处理的脑电数据。

1. 如果想对伪迹处理后的数据进行验证，需要将下载后的derivatives文件夹放到FPSD-CNN-MAIN文件夹下，而后执行代码 data_split-AD_CN或者 data_split-CN_FTD来对数据进行划分。

2. 人工想对未经过伪迹处理的数据进行验证，需要将derivatives文件夹外的文件夹全部复制，移入到新建的文件夹中，新建文件夹的命名可以为derivatives或者任意，但是如果任意命名需要修改data_split-AD_CN或者 data_split-CN_FTD代码中的source_directory = 'derivatives'  # 替换为你的源文件夹路径，为你的目标代码