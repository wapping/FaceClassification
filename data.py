# -*- coding: utf-8 -*-
"""
Description :
Authors     : Wapping
CreateDate  : 2021/8/6
"""
import os
import cv2
import logging
import numpy as np
from paddle.io import Dataset
import scipy.ndimage as ndi
from scipy.io import loadmat
from scipy.misc import imread, imresize
from random import shuffle
from utils import preprocess_input


logging.basicConfig(format="%(levelname)s - %(message)s", level=logging.INFO)


class FaceDataset(Dataset):
    def __init__(self, data, img_path_prefix="", img_size=(64, 64),
                 do_transform=False,
                 saturation_var=0.5,
                 brightness_var=0.5,
                 contrast_var=0.5,
                 lighting_std=0.5,
                 horizontal_flip_probability=0.5,
                 vertical_flip_probability=0.5,
                 do_random_crop=False,
                 grayscale=False,
                 zoom_range=(0.75, 1.25),
                 translation_factor=.3,
                 data_format='NCHW',
                 read_img_at_once=False):

        super(FaceDataset, self).__init__()
        self.data = data
        self.img_path_prefix = img_path_prefix
        self.img_size = img_size
        self.do_transform = do_transform
        self.saturation_var = saturation_var
        self.brightness_var = brightness_var
        self.contrast_var = contrast_var
        self.lighting_std = lighting_std
        self.horizontal_flip_probability = horizontal_flip_probability
        self.vertical_flip_probability = vertical_flip_probability
        self.do_random_crop = do_random_crop
        self.grayscale = grayscale
        self.zoom_range = zoom_range
        self.translation_factor = translation_factor
        self.data_format = data_format
        self.color_jitter = []
        if saturation_var:
            self.saturation_var = saturation_var
            self.color_jitter.append(self.saturation)
        if brightness_var:
            self.brightness_var = brightness_var
            self.color_jitter.append(self.brightness)
        if contrast_var:
            self.contrast_var = contrast_var
            self.color_jitter.append(self.contrast)
        self.n_samples = len(self.data)
        self.images = []
        self.read_img_at_once = read_img_at_once
        if read_img_at_once:
            self.read_resize_images()

    def __getitem__(self, index):
        if self.read_img_at_once:
            return self._getitem_v2(index)
        else:
            return self._getitem(index)

    def _getitem(self, index):
        item = self.data[index]
        img_path, label = item[0], item[1]
        if self.img_path_prefix:
            img_path = os.path.join(self.img_path_prefix, img_path)

        image_array = imread(img_path)

        is_gray = True if len(image_array.shape) == 2 else False

        if is_gray and not self.grayscale:
            logging.warning(f"'{img_path}' is a gray image, but 'self.grayscale' is False.")
            image_array = np.expand_dims(image_array, -1).repeat(3, -1)

        image_array = imresize(image_array, self.img_size)

        if self.do_random_crop and not is_gray:
            image_array = self._do_random_crop(image_array)

        image_array = image_array.astype('float32')

        if self.do_transform and not is_gray:
            image_array = self.transform(image_array)[0]

        if self.grayscale:
            if not is_gray:
                image_array = cv2.cvtColor(
                    image_array.astype('uint8'),
                    cv2.COLOR_RGB2GRAY).astype('float32')
            image_array = np.expand_dims(image_array, -1)

        image_array = self.preprocess_images(image_array)

        if self.data_format == 'NCHW':
            image_array = np.transpose(image_array, (2, 1, 0))
        return image_array, label

    def _getitem_v2(self, index):
        item = self.data[index]
        img_path, label = item[0], item[1]
        if self.img_path_prefix:
            img_path = os.path.join(self.img_path_prefix, img_path)

        image_array = self.images[index]

        is_gray = True if len(image_array.shape) == 2 else False

        if is_gray and not self.grayscale:
            logging.warning(f"'{img_path}' is a gray image, but 'self.grayscale' is False.")
            image_array = np.expand_dims(image_array, -1).repeat(3, -1)

        if self.do_random_crop and not is_gray:
            image_array = self._do_random_crop(image_array)

        image_array = image_array.astype('float32')

        if self.do_transform and not is_gray:
            image_array = self.transform(image_array)[0]

        if self.grayscale:
            if not is_gray:
                image_array = cv2.cvtColor(
                    image_array.astype('uint8'),
                    cv2.COLOR_RGB2GRAY).astype('float32')
            image_array = np.expand_dims(image_array, -1)

        image_array = self.preprocess_images(image_array)

        if self.data_format == 'NCHW':
            image_array = np.transpose(image_array, (2, 1, 0))
        return image_array, label

    def read_resize_images(self):
        self.images = []
        for item in self.data:
            img_path, label = item[0], item[1]

            if self.img_path_prefix:
                img_path = os.path.join(self.img_path_prefix, img_path)

            image_array = imread(img_path)

            image_array = imresize(image_array, self.img_size)

            self.images.append(image_array)

    def __len__(self):
        return self.n_samples

    def _do_random_crop(self, image_array):
        """IMPORTANT: random crop only works for classification since the
        current implementation does no transform bounding boxes"""
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                         self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                         crop_matrix, offset=offset, order=0, mode='nearest',
                         cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def do_random_rotation(self, image_array):
        """IMPORTANT: random rotation only works for classification since the
        current implementation does no transform bounding boxes"""
        height = image_array.shape[0]
        width = image_array.shape[1]
        x_offset = np.random.uniform(0, self.translation_factor * width)
        y_offset = np.random.uniform(0, self.translation_factor * height)
        offset = np.array([x_offset, y_offset])
        scale_factor = np.random.uniform(self.zoom_range[0],
                                         self.zoom_range[1])
        crop_matrix = np.array([[scale_factor, 0],
                                [0, scale_factor]])

        image_array = np.rollaxis(image_array, axis=-1, start=0)
        image_channel = [ndi.interpolation.affine_transform(image_channel,
                         crop_matrix, offset=offset, order=0, mode='nearest',
                         cval=0.0) for image_channel in image_array]

        image_array = np.stack(image_channel, axis=0)
        image_array = np.rollaxis(image_array, 0, 3)
        return image_array

    def _gray_scale(self, image_array):
        return image_array.dot([0.299, 0.587, 0.114])

    def saturation(self, image_array):
        gray_scale = self._gray_scale(image_array)
        alpha = 2.0 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = (alpha * image_array + (1 - alpha) *
                       gray_scale[:, :, None])
        return np.clip(image_array, 0, 255)

    def brightness(self, image_array):
        alpha = 2 * np.random.random() * self.brightness_var
        alpha = alpha + 1 - self.saturation_var
        image_array = alpha * image_array
        return np.clip(image_array, 0, 255)

    def contrast(self, image_array):
        gray_scale = (self._gray_scale(image_array).mean() *
                      np.ones_like(image_array))
        alpha = 2 * np.random.random() * self.contrast_var
        alpha = alpha + 1 - self.contrast_var
        image_array = image_array * alpha + (1 - alpha) * gray_scale
        return np.clip(image_array, 0, 255)

    def lighting(self, image_array):
        covariance_matrix = np.cov(image_array.reshape(-1, 3) /
                                   255.0, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        noise = np.random.randn(3) * self.lighting_std
        noise = eigen_vectors.dot(eigen_values * noise) * 255
        image_array = image_array + noise
        return np.clip(image_array, 0, 255)

    def horizontal_flip(self, image_array, box_corners=None):
        if np.random.random() < self.horizontal_flip_probability:
            image_array = image_array[:, ::-1]
            if box_corners is not None:
                box_corners[:, [0, 2]] = 1 - box_corners[:, [2, 0]]
        return image_array, box_corners

    def vertical_flip(self, image_array, box_corners=None):
        if np.random.random() < self.vertical_flip_probability:
            image_array = image_array[::-1]
            if box_corners is not None:
                box_corners[:, [1, 3]] = 1 - box_corners[:, [3, 1]]
        return image_array, box_corners

    def transform(self, image_array, box_corners=None):
        shuffle(self.color_jitter)
        for jitter in self.color_jitter:
            image_array = jitter(image_array)

        if self.lighting_std:
            image_array = self.lighting(image_array)

        if self.horizontal_flip_probability > 0:
            image_array, box_corners = self.horizontal_flip(image_array,
                                                            box_corners)

        if self.vertical_flip_probability > 0:
            image_array, box_corners = self.vertical_flip(image_array,
                                                          box_corners)
        return image_array, box_corners

    def preprocess_images(self, image_array):
        return preprocess_input(image_array)


def load_imdb(data_path):
    face_score_threshold = 3
    dataset = loadmat(data_path)
    image_names_array = dataset['imdb']['full_path'][0, 0][0]
    gender_classes = dataset['imdb']['gender'][0, 0][0]
    face_score = dataset['imdb']['face_score'][0, 0][0]
    second_face_score = dataset['imdb']['second_face_score'][0, 0][0]
    face_score_mask = face_score > face_score_threshold
    second_face_score_mask = np.isnan(second_face_score)
    unknown_gender_mask = np.logical_not(np.isnan(gender_classes))
    mask = np.logical_and(face_score_mask, second_face_score_mask)
    mask = np.logical_and(mask, unknown_gender_mask)
    image_names_array = image_names_array[mask]
    gender_classes = gender_classes[mask].tolist()
    image_names = []
    for image_name_arg in range(image_names_array.shape[0]):
        image_name = image_names_array[image_name_arg][0]
        image_names.append(image_name)
    data = [(k, int(v)) for k, v in zip(image_names, gender_classes)]
    return data


def split_imdb_data(data, validation_split=.2, do_shuffle=False):
    if do_shuffle:
        shuffle(data)
    else:
        data = sorted(data, key=lambda x: x[0])

    training_split = 1 - validation_split
    num_train = int(training_split * len(data))
    train_set = data[:num_train]
    val_set = data[num_train:]
    return train_set, val_set
