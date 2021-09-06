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
"""
Description: An implementation of MiniXception proposed in the paper `Real-time Convolutional Neural Networks for Emotion and Gender Classification` (https://arxiv.org/pdf/1710.07557v1.pdf).
"""
import paddle
from paddle import nn


class DepthWiseSepConv(nn.Layer):
    """Depth wise separable convolution layer."""
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: The number of channels of the input of the layer.
            out_channels: The number of channels of the output of the layer.
        """
        super(DepthWiseSepConv, self).__init__()
        self.depth_wise = nn.Conv2D(in_channels, in_channels, kernel_size=3, padding='same', groups=in_channels,
                                    bias_attr=False)
        self.point_wise = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False)

    def forward(self, x):
        """Forward.
        Args:
            x: The input data (images), A tensor with shape (N, C, H, W), C: n_channels.
        Return:
            out: The output of the model, A tensor with shape (N, C', H', W').
        """
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out


class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels: The number of channels of the input of the layer.
            out_channels: The number of channels of the output of the layer.
        """
        super(ResBlock, self).__init__()
        self.max_pool = nn.MaxPool2D(3, 2, padding=1)

        self.sep_conv1 = nn.Sequential(
            DepthWiseSepConv(in_channels, out_channels),
            nn.BatchNorm2D(out_channels),
            nn.ReLU()
        )

        self.sep_conv2 = nn.Sequential(
            DepthWiseSepConv(out_channels, out_channels),
            nn.BatchNorm2D(out_channels)
        )

        self.conv = nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=2, padding='same', bias_attr=False),
            nn.BatchNorm2D(out_channels)
        )

    def forward(self, x):
        """Forward.
        Args:
            x: The input data (images), A tensor with shape (N, C, H, W), C: n_channels.
        Return:
            out: The output of the model, A tensor with shape (N, C', H', W').
        """
        path1 = self.sep_conv1(x)
        path1 = self.sep_conv2(path1)
        path1 = self.max_pool(path1)
        path2 = self.conv(x)
        out = path1 + path2
        return out


class MiniXception(nn.Layer):
    """The MiniXception structure."""
    def __init__(self, n_classes=2, in_channels=1):
        """
        Args:
            n_classes: The number of classes.
            in_channels: The number of channels of the input image of the model.
        """
        super(MiniXception, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2D(self.in_channels, 8, 3, 1, bias_attr=False),
            nn.BatchNorm2D(8),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2D(8, 8, 3, 1, bias_attr=False),
            nn.BatchNorm2D(8),
            nn.ReLU()
        )

        self.res_blocks = nn.Sequential(
            ResBlock(8, 16),
            ResBlock(16, 32),
            ResBlock(32, 64),
            ResBlock(64, 128),
        )

        self.conv3 = nn.Conv2D(128, self.n_classes, 3)

        self.global_avg_pool = nn.AdaptiveAvgPool2D(output_size=(1, 1))

    def forward(self, x):
        """Forward.
        Args:
            x: The input data (images), A tensor with shape (N, C, H, W), C: n_channels.
        Returns:
            out: The output of the model, A tensor with shape (N, n_classes).
        """
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.res_blocks(out)

        out = self.conv3(out)

        out = self.global_avg_pool(out)

        out = out.squeeze((2, 3))

        return out


def mini_xception_test():
    data = paddle.rand((8, 1, 64, 64))
    model = MiniXception()
    out = model(data)
    print(f"input shape: {data.shape}, output shape: {out.shape}")


if __name__ == '__main__':
    ...
    mini_xception_test()
