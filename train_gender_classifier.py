# -*- coding: utf-8 -*-
"""
Description :
Authors     : lihp
CreateDate  : 2021/8/6
"""
import os
import sys
import yaml
import logging
import numpy as np
import random
import paddle
from tqdm import tqdm
from net import MiniXception, SimpleCNN
from data import load_imdb, split_imdb_data
from data import FaceDataset
from visualdl import LogWriter
from paddle.io import DataLoader
from paddle import distributed as dist

logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


def train():
    # Loading dataset
    logging.info(f"Loading dataset ...")

    data = load_imdb(os.path.join(data_args['imdb_dir'], 'imdb.mat'))
    train_set, val_set = split_imdb_data(data, args['validation_split'])

    train_set = FaceDataset(train_set, img_path_prefix=data_args['imdb_dir'], grayscale=data_args['grayscale'],
                            do_transform=True,
                            do_random_crop=data_args['do_rand_crop'],
                            translation_factor=data_args['translation_factor'],
                            vertical_flip_probability=data_args['vertical_flip_probability'],
                            read_img_at_once=data_args['read_img_at_once'],
                            img_size=data_args['img_size'],
                            )
    val_set = FaceDataset(val_set, img_path_prefix=data_args['imdb_dir'], grayscale=data_args['grayscale'],
                          read_img_at_once=data_args['read_img_at_once'],
                          img_size=data_args['img_size'],
                          )
    logging.info(
        f"The numbers of samples of training and validating sets are {train_set.__len__()} and {val_set.__len__()}.")

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=4)

    # For multi-GPUs training
    dist.init_parallel_env()

    # Building model, optimizer and loss function
    if args['model_name'] == "MiniXception":
        model = MiniXception()
    else:
        model = SimpleCNN(in_channels=args['in_channels'])

    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.001, factor=0.5, patience=50, verbose=True, epsilon=1e-6)
    optimizer = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters(),
                                      # weight_decay=0.001,
                                      )
    loss_fn = paddle.nn.CrossEntropyLoss()

    # Restore parameters
    if args['restore']:
        model = restore_params(model, args['model_state_dict'])
        optimizer = restore_params(optimizer, args['opt_state_dict'])
        epoch = int(args['model_state_dict'].split('-')[1]) + 1
    else:
        epoch = 1

    # For multi-GPUs training
    model = paddle.DataParallel(model)

    # For visualDL
    train_logger = LogWriter(logdir=os.path.join(args['logdir'], 'train'))
    val_logger = LogWriter(logdir=os.path.join(args['logdir'], 'val'))

    # Starting training
    max_val_acc = 0.95
    for epoch in range(epoch, args['epochs'] + 1):
        logging.info("=" * 50 + f"Epoch {epoch}" + "=" * 50)

        # Training
        n_samples, sum_acc, sum_loss = 0, 0., 0.
        model.train()
        for step, batch in enumerate(train_loader, 1):
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

            if step % 100 == 0:
                logging.info(f"Step {step}, loss {loss.item()}, acc {acc.item()}")
                # train_logger.add_scalar('loss', loss.item(), (epoch - 1) * len(train_loader) + step)
                # train_logger.add_scalar('acc', acc.item(), (epoch - 1) * len(train_loader) + step)
                # train_logger.add_scalar('lr', optimizer.get_lr(), (epoch - 1) * len(train_loader) + step)

        loss, acc = sum_loss / n_samples, sum_acc / n_samples

        train_logger.add_scalar('loss', loss, epoch)
        train_logger.add_scalar('acc', acc, epoch)
        train_logger.add_scalar('lr', optimizer.get_lr(), epoch)

        # Validation
        n_samples, sum_acc, sum_loss = 0, 0., 0.
        model.eval()
        for batch in val_loader():
            inputs, labels = batch[0], batch[1]

            pred = model(inputs)

            loss = loss_fn(pred, labels)

            acc = paddle.metric.accuracy(pred, labels.unsqueeze(1))

            n = len(inputs)

            sum_loss += loss.item() * n
            sum_acc += acc.item() * n
            n_samples += n

        loss, acc = sum_loss / n_samples, sum_acc / n_samples
        logging.info(f"Epoch {epoch}, Val loss {loss}, Val acc {acc}")
        # val_logger.add_scalar('loss', loss, epoch * len(train_loader))
        # val_logger.add_scalar('acc', acc, epoch * len(train_loader))
        val_logger.add_scalar('loss', loss, epoch)
        val_logger.add_scalar('acc', acc, epoch)

        scheduler.step(loss, epoch)

        if acc > max_val_acc:
            logging.info(f"Epoch {epoch}, Output model with val acc {acc} to {args['model_dir']}")
            paddle.save(model.state_dict(),
                        os.path.join(args['model_dir'], f"{args['model_name']}-{epoch}-{acc}.params"))
            paddle.save(optimizer.state_dict(), os.path.join(args['model_dir'], f"{args['model_name']}-{epoch}.opt"))
            max_val_acc = acc
            if acc >= 0.96:
                logging.info("Finished training.")
                break


def restore_params(model, model_path):
    logging.info(f"Restoring parameters from {model_path}")
    model_state_dict = paddle.load(model_path)
    model.set_state_dict(model_state_dict)
    return model


def validate():
    logging.info(f"Loading dataset ...")

    data = load_imdb(os.path.join(data_args['imdb_dir'], 'imdb.mat'))

    _, val_set = split_imdb_data(data, args['validation_split'])

    val_set = FaceDataset(val_set, img_path_prefix=data_args['imdb_dir'], grayscale=data_args['grayscale'],
                          img_size=data_args['img_size']
                          )

    logging.info(
        f"The number of validation samples is {val_set.__len__()}.")

    val_loader = DataLoader(val_set, batch_size=args['batch_size'], num_workers=4)

    if args['model_name'] == "MiniXception":
        model = MiniXception()
    else:
        model = SimpleCNN(in_channels=args['in_channels'])

    model_state_dict = paddle.load(args['model_state_dict'])
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

    logging.info(f"Val loss {loss}, Val acc {acc}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


if __name__ == '__main__':
    conf_path = 'conf.yaml'

    if len(sys.argv) > 1:
        conf_path = sys.argv[1]

    args = yaml.load(open(conf_path).read())

    data_args = args['dataset']

    set_seed(args['seed'])

    if args.get('mode', 'train') == 'val':
        validate()
    else:
        train()

