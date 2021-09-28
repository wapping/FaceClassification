English | [简体中文](./README_cn.md) 

[TOC]
# 一、Introduction
An implementation of the face (emotion and gender) classification models (`MiniXception`。`SimpleCNN`) proposed in paper [`Real-time Convolutional Neural Networks for Emotion and Gender Classification`](https://arxiv.org/pdf/1710.07557v1.pdf)  with PaddlePaddle. `SimpleCNN` is a standard fully-convolutional neural network composed of 9 convolution layers, ReLUs, Batch Normalization and Global Average Pooling.  `MiniXception` replaces the convolution layers with depth-wise separable convolutions and residual modules. 

# 二、Accuracy

We trained the models with `imdb_crop` dataset in gender classification task and each model obtains an accuracy of 96%.

| Model | Accuracy | Input shape |
|  :---  | ----  | ----  |
| SimpleCNN | 96.00% | (48, 48, 3) |
| MiniXception | 96.01% | (64, 64, 1) |

# 三、Dataset

We trained and tested models on dataset [imdb_crop](https://pan.baidu.com/s/1xdFxhxcnO_5WyQh7URWMQA) (the password is `mu2h`).  The dataset can be also download from [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). First, download and uncompress the dataset. Then, edit configuration files `config/simple_conf.yaml` and `config/min_conf.yaml` of models `SimpleCNN` and `MiniXception`. Set `imdb_dir` to be the path to the dataset. `imdb_dir` s should be the same in training and test stages. You don't need to split the dataset into training and test set, because the python scripts will do that. The dataset will be split in the manner of that proposed in the [paper](https://github.com/oarriaga/face_classification). That is, sorts the images by file names and considers the front 80% as training set and the rear 20% as test set. 

# 四、Environment

```
scipy==1.2.1
paddlepaddle==2.1.2
numpy==1.20.1
opencv-python==3.4.10.37
pyyaml~=5.4.1
visualdl~=2.2.0
tqdm~=4.62.0
```

# 五、Quick Start

## Step1: Clone

```shell
# clone this repo
git clone https://github.com/wapping/FaceClassification.git
cd FaceClassification
```

## Step2: Train

Edit the configuration file for your own and run the command like

```shell
python train.py path_to_conf
```
For example
```shell
python train.py ./config/simple_conf.yaml
```


## Step3: Test

Edit the configuration file for your own and run the command like

```shell
python eval.py path_to_conf
```

Just wait for the results.

# 六、Code Structure and Explanation
## 6.1 Code Structure

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



## 6.2 Parameter Explanation

- train.py

  `--conf_path`: optional, the path to the configuration file, `config/conf.yaml` by default.

  `--model_name`: optional, the model name. If given, it will replace `model name` in the configuration file.

- eval.py
`--conf_path`: optional, the path to the configuration file, `config/conf.yaml` by default.

  `--model_name`: optional, the model name. If given, it will replace `model name` in the configuration file.
  `--model_state_dict`: optional, the path to the model. If given, it will replace `model_state_dict` in the configuration file.

# 七、Model Infomation
| Field | Content |
|  :---  | ----  |
| Author | Huaping Li、Xiaoqian Song |
| Date | 2021.09 |
| Framework version | paddlepaddle 2.1.2 |
| Application scenarios | Face classification |
| Supported hardware | CPU、GPU |



