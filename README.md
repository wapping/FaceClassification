# About
An implementation of paper [`Real-time Convolutional Neural Networks for Emotion and Gender Classification`](https://arxiv.org/pdf/1710.07557v1.pdf) with PaddlePaddle.  

# Requirements

```
scipy==1.2.1
paddlepaddle==2.1.2
numpy==1.21.1
opencv-python==3.4.10.37
pyyaml~=5.4.1
visualdl~=2.2.0
```

# Data Preparation

We trained the models with the same dataset [imdb_crop](https://pan.baidu.com/s/1xdFxhxcnO_5WyQh7URWMQA) (password is `mu2h`). The dataset can be also downloaded from [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/). After data download and uncompression, edit the configuration file (`conf.yaml` or your own file), setting the data directory `imdb_dir`.



# Training

Just run

```shell
python train_gender_classfifier.py path_to_your_configration
```

like
```shell
python train_gender_classfifier.py ./conf.yaml
```

`path_to_your_configration` is optional. The default path is `./conf.yaml`.












