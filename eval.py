# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import logging
import paddle
from tqdm import tqdm
from models.simple_cnn import SimpleCNN
from models.mini_xception import MiniXception
from data.dataset import load_imdb, split_imdb_data
from data.dataset import FaceDataset
from paddle.io import DataLoader
from argparse import ArgumentParser
from config.confg import parse_args


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def main():
    logging.info("Loading dataset ...")
    data = load_imdb(os.path.join(data_args.imdb_dir, 'imdb.mat'))
    _, val_set = split_imdb_data(data, args.validation_split)
    val_set = FaceDataset(val_set, img_path_prefix=data_args.imdb_dir, grayscale=data_args.grayscale,
                          img_size=data_args.img_size)
    logging.info(f"The number of validation samples is {val_set.__len__()}.")

    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=4)

    if args.model_name == "MiniXception":
        model = MiniXception(n_classes=args.n_classes, in_channels=args.in_channels)
    else:
        model = SimpleCNN(n_classes=args.n_classes, in_channels=args.in_channels)

    model_state_dict = paddle.load(args.model_state_dict)
    model.set_state_dict(model_state_dict)

    loss_fn = paddle.nn.CrossEntropyLoss()

    n_samples, sum_acc, sum_loss = 0, 0., 0.
    model.eval()
    for batch in tqdm(val_loader()):
        inputs, labels = batch[0], batch[1]

        pred = model(inputs)

        loss = loss_fn(pred, labels)

        acc = paddle.metric.accuracy(pred, labels.unsqueeze(1))

        n = len(inputs)

        sum_loss += loss.item() * n
        sum_acc += acc.item() * n
        n_samples += n

    loss, acc = sum_loss / n_samples, sum_acc / n_samples

    logging.info(f"Simples: {n_samples}, Val loss: {loss}, Val accuracy: {acc}")


if __name__ == '__main__':
    parser = ArgumentParser("Eval parameters")
    parser.add_argument('--conf_path', '-c', type=str, default='config/conf.yaml', help='Path to the config.')
    parser.add_argument('--model_name', '-m', type=str, choices=['MiniXception', 'SimpleCNN'], help='Choose a model.')
    parser.add_argument('--model_state_dict', '-msd', type=str, help='Path to the model parameters.')

    args = parse_args(parser)

    data_args = args.dataset

    main()

