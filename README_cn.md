[English](./README.md) | 简体中文

[TOC]
# 一、简介
通过PaddlePaddle框架复现了论文 [`Real-time Convolutional Neural Networks for Emotion and Gender Classification`](https://arxiv.org/pdf/1710.07557v1.pdf) 中提出的两个人脸（性别、表情）分类模型，分别是`SimpleCNN`和`MiniXception`。`SimpleCNN`由9个卷积层、ReLU、Batch Normalization和Global Average Pooling组合成，`MiniXception`结合了深度可分离卷积和残差模块，两者都是全卷积神经网络。

# 二、复现精度

利用 `imdb_crop`数据集训练模型，进行人脸性别分类，准确率均达到96%。

| 模型 | 准确率 | 输入尺寸 |
|  :---  | ----  | ----  |
| SimpleCNN | 96.00% | (48, 48, 3) |
| MiniXception | 96.01% | (64, 64, 1) |

# 三、数据集

我们在数据集[imdb_crop](https://pan.baidu.com/s/1xdFxhxcnO_5WyQh7URWMQA) (密码 `mu2h`)上训练和测试模型，数据集也可以在[这里](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)下载。下载和解压数据后，不用对数据再做别的处理了，编辑配置文件`config/simple_conf.yaml`和`config/min_conf.yaml`，两者分别是模型`SimpleCNN`和`MiniXception`的配置文件，把 `imdb_dir`设置成数据集所在的目录。训练和测试阶段的 `imdb_dir`应该是一致的。不用划分训练集和测试集，程序会自动划分，即使你不训练只测试。我们采取的数据集划分方式和论文[作者的](https://github.com/oarriaga/face_classification)一样，先根据文件名对图片进行排序，前80%为训练集，后20%为测试集。

# 四、环境依赖

```
scipy==1.2.1
paddlepaddle==2.1.2
numpy==1.20.1
opencv-python==3.4.10.37
pyyaml~=5.4.1
visualdl~=2.2.0
tqdm~=4.62.0
```

# 五、快速开始

## Step1: clone

```shell
# clone this repo
git clone https://github.com/wapping/FaceClassification.git
cd FaceClassification
```

## Step2: 训练

根据实际情况配置文件，然后执行命令

```shell
python train.py path_to_conf
```
比如
```shell
python train.py ./config/simple_conf.yaml
```


## Step3: 测试

根据实际情况配置文件，然后执行命令

```shell
python eval.py path_to_conf
```

等结果就行了。

# 六、代码结构与详细说明
## 6.1 代码结构

```
|____config
| |____conf.yaml
| |____confg.py
| |____simple_conf.yaml
| |____mini_conf.yaml
|____data
| |____dataset.py
|____models
| |____simple_cnn.py
| |____mini_xception.py
|____train.py
|____eval.py
```



## 6.2 参数说明

- train.py

  `--conf_path`: 可选，配置文件的路径，默认是`config/conf.yaml`。

  `--model_name`: 可选，模型的名称，如果提供，则替换配置文件里的`model_name`。

- eval.py
`--conf_path`: 可选，配置文件的路径，默认是`config/conf.yaml`。

  `--model_name`: 可选，模型的名称，如果提供，则替换配置文件里的`model_name`。
  `--model_state_dict`: 可选，模型文件路径，如果提供，则替换配置文件里的`model_state_dict`。

# 七、模型信息
| 字段 | 内容 |
|  :---  | ----  |
| 发布者 | 李华平、宋晓倩 |
| 时间 | 2021.09 |
| 框架版本 | paddlepaddle 2.1.2 |
| 应用场景 | 人脸性别分类 |
| 支持硬件 | CPU、GPU |



