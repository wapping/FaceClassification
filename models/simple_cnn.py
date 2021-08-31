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

import paddle
from paddle import nn


class SimpleCNN(nn.Layer):
    def __init__(self, n_classes=2, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2D(self.in_channels, 16, 7, 1, padding='same'),
            nn.BatchNorm2D(16),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2D(16, 16, 7, 1, padding='same'),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2D(16, 32, 5, 1, padding='same'),
            nn.BatchNorm2D(32),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2D(32, 32, 5, 1, padding='same'),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 1, padding='same'),
            nn.BatchNorm2D(64),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, padding='same'),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2D(64, 128, 3, 1, padding='same'),
            nn.BatchNorm2D(128),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2D(128, 128, 3, 1, padding='same'),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2D(128, 256, 3, 1, padding='same'),
            nn.BatchNorm2D(256),
        )

        self.global_avg_pool = nn.Sequential(
            nn.Conv2D(256, self.n_classes, 3, 1, padding='same'),
            nn.AdaptiveAvgPool2D(output_size=(1, 1)),
        )

    def forward(self, x):
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.conv5(out)

        out = self.conv6(out)

        out = self.conv7(out)

        out = self.conv8(out)

        out = self.conv9(out)

        out = self.global_avg_pool(out)

        out = out.squeeze((2, 3))

        return out


def simple_cnn_test():
    data = paddle.rand((8, 3, 64, 64))
    model = SimpleCNN(in_channels=3)
    out = model(data)
    print(f"input shape: {data.shape}, output shape: {out.shape}")


if __name__ == '__main__':
    ...
    simple_cnn_test()
