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
import numpy as np
import random
import paddle
from tqdm import tqdm
from models.simple_cnn import SimpleCNN
from models.mini_xception import MiniXception
from data.dataset import load_imdb, split_imdb_data
from data.dataset import FaceDataset
from visualdl import LogWriter
from argparse import ArgumentParser
from paddle.io import DataLoader
from paddle import distributed as dist
from config.confg import parse_args

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def train():
    # Loading the dataset
    logging.info(f"Loading the dataset ...")
    data = load_imdb(os.path.join(data_args.imdb_dir, 'imdb.mat'))
    train_set, val_set = split_imdb_data(data, args.validation_split)
    train_set = FaceDataset(train_set, img_path_prefix=data_args.imdb_dir, grayscale=data_args.grayscale,
                            do_transform=True,
                            do_random_crop=data_args.do_rand_crop,
                            translation_factor=data_args.translation_factor,
                            vertical_flip_probability=data_args.vertical_flip_probability,
                            read_img_at_once=data_args.read_img_at_once,
                            img_size=data_args.img_size,
                            )
    val_set = FaceDataset(val_set, img_path_prefix=data_args.imdb_dir, grayscale=data_args.grayscale,
                          read_img_at_once=data_args.read_img_at_once,
                          img_size=data_args.img_size,
                          )
    logging.info(
        f"The numbers of samples of training and validation sets are {train_set.__len__()} and {val_set.__len__()}.")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    # For multi-GPUs training
    dist.init_parallel_env()

    # Building model, optimizer and loss function
    if args.model_name == "MiniXception":
        model = MiniXception(args.n_classes, args.in_channels)
    else:
        model = SimpleCNN(args.n_classes, args.in_channels)
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.001, factor=0.5, patience=50, verbose=True,
                                                    epsilon=1e-6)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()

    # Restore parameters
    if args.restore:
        model = restore_params(model, args.model_state_dict)
        optimizer = restore_params(optimizer, args.opt_state_dict)
        epoch = int(args.model_state_dict.split('-')[1]) + 1
    else:
        epoch = 1

    # For multi-GPUs training
    model = paddle.DataParallel(model)

    # For visualDL
    train_logger = LogWriter(logdir=os.path.join(args.logdir, 'train'))
    val_logger = LogWriter(logdir=os.path.join(args.logdir, 'val'))

    # Starting training
    saved_params = []
    max_val_acc = 0.95
    for epoch in range(epoch, args['epochs'] + 1):
        logging.info("=" * 50 + f"Epoch {epoch}" + "=" * 50)

        # Training
        n_samples, sum_acc, sum_loss = 0, 0., 0.
        model.train()
        with tqdm(train_loader) as t:
            for step, batch in enumerate(t, 1):
                inputs, labels = batch[0], batch[1]
                pred = model(inputs)
                loss = loss_fn(pred, labels)
                acc = paddle.metric.accuracy(pred, labels.unsqueeze(1))
                n = len(inputs)
                sum_loss += loss.item() * n
                sum_acc += acc.item() * n
                n_samples += n
                loss.backward()
                optimizer.step()
                optimizer.clear_grad()
                t.set_postfix(train_loss=loss.item(), train_acc=acc.item())
            # if step % 100 == 0:
            #     logging.info(f"Step {step}, loss {loss.item()}, acc {acc.item()}")

        loss, acc = sum_loss / n_samples, sum_acc / n_samples
        train_logger.add_scalar('loss', loss, epoch)
        train_logger.add_scalar('acc', acc, epoch)
        train_logger.add_scalar('lr', optimizer.get_lr(), epoch)
        logging.info(f"Epoch {epoch}, train loss {loss}, train acc {acc}, lr {optimizer.get_lr()}.")

        # Validation
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
        val_logger.add_scalar('loss', loss, epoch)
        val_logger.add_scalar('acc', acc, epoch)
        logging.info(f"Epoch {epoch}, val loss {loss}, val acc {acc}")
        scheduler.step(loss, epoch)

        if acc > max_val_acc or epoch >= args.epochs:
            logging.info(f"Epoch {epoch}, Output model with val acc {acc} to {args.model_dir}")
            model_path = os.path.join(args.model_dir, f"{args.model_name}-{epoch}-{acc}.params")
            opt_path = os.path.join(args.model_dir, f"{args.model_name}-{epoch}-{acc}.opt")
            paddle.save(model.state_dict(), model_path)
            paddle.save(optimizer.state_dict(), opt_path)
            if args.just_keep_the_best:
                for p in saved_params:
                    os.remove(p)
            saved_params = [model_path, opt_path]
            max_val_acc = acc
            if acc >= 0.96:
                logging.info("Finished training.")
                break
    logging.info("Finished training.")


def restore_params(model, model_path):
    logging.info(f"Restoring parameters from {model_path}")
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    return model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


if __name__ == '__main__':
    parser = ArgumentParser("Training parameters")
    parser.add_argument('--conf_path', '-c', type=str, default='config/conf.yaml', help='Path to the config.')
    parser.add_argument('--model_name', '-m', type=str, choices=['MiniXception', 'SimpleCNN'],
                        help='Choose a model to train.')

    args = parse_args(parser)

    data_args = args.dataset

    set_seed(args.seed)

    train()
