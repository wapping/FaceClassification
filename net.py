# -*- coding: utf-8 -*-
"""
Description :
Authors     : Wapping
CreateDate  : 2021/8/5
"""
import paddle
from paddle import nn


class DepthWiseSepConv(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super(DepthWiseSepConv, self).__init__()
        self.depth_wise = nn.Conv2D(in_channels, in_channels, kernel_size=3, padding='same', groups=in_channels, bias_attr=False)
        self.point_wise = nn.Conv2D(in_channels, out_channels, kernel_size=1, bias_attr=False)

    def forward(self, x):
        out = self.depth_wise(x)
        out = self.point_wise(out)
        return out


class ResBlock(nn.Layer):
    def __init__(self, in_channels, out_channels):
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
        """...
        """
        path1 = self.sep_conv1(x)
        path1 = self.sep_conv2(path1)
        path1 = self.max_pool(path1)
        path2 = self.conv(x)
        out = path1 + path2
        return out


class MiniXception(nn.Layer):
    def __init__(self, n_classes=2, in_channels=1):
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
        out = self.conv1(x)

        out = self.conv2(out)

        out = self.res_blocks(out)

        out = self.conv3(out)

        out = self.global_avg_pool(out)

        out = out.squeeze((2, 3))

        return out


class SimpleCNN(nn.Layer):
    def __init__(self, n_classes=2, in_channels=1):
        super(SimpleCNN, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels

        self.conv1 = nn.Sequential(
            nn.Conv2D(self.in_channels, 16, 7, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(16),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2D(16, 16, 7, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2D(16, 32, 5, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(32),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2D(32, 32, 5, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(64),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2D(64, 64, 3, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(64),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv7 = nn.Sequential(
            nn.Conv2D(64, 128, 3, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(128),
        )

        self.conv8 = nn.Sequential(
            nn.Conv2D(128, 128, 3, 1, padding='same', bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.AvgPool2D(kernel_size=(2, 2), padding='same'),
            nn.Dropout(.25),
        )

        self.conv9 = nn.Sequential(
            nn.Conv2D(128, 256, 3, 1, padding='same', bias_attr=False),
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


def mini_xception_test():
    data = paddle.rand((8, 1, 64, 64))
    model = MiniXception()
    out = model(data)
    print(f"input shape: {data.shape}, output shape: {out.shape}")


def simple_cnn_test():
    data = paddle.rand((8, 1, 64, 64))
    model = SimpleCNN()
    out = model(data)
    print(f"input shape: {data.shape}, output shape: {out.shape}")


if __name__ == '__main__':
    ...
    simple_cnn_test()
    mini_xception_test()
