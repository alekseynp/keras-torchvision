from keras.models import Model
from keras.layers import Input, Activation, Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Cropping2D
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization

from math import ceil, floor

import functools


def sequential(*functions):
    """
    Achieves the same as torch.nn.Sequential

    :param functions: list of functions
    :return: single function composing those in the list in the order that they appeared
    """
    return functools.reduce(lambda f, g: lambda x: f(g(x)), list(reversed(functions)), lambda x: x)


def same_pad_like_pytorch(operation, operation_size):
    def build(tensor):
        input_size = int(tensor.shape[1])
        out_size = input_size // 2
        width_needed_for_valid_conv = 1 + 2 * (out_size - 1) + 2 * (operation_size // 2)
        needed_padding = width_needed_for_valid_conv - input_size
        padding_left = ceil(needed_padding / 2.)
        padding_right = floor(needed_padding / 2.)

        if needed_padding < 0:
            # This happens with the 1x1 stride 2 convs
            padder = Cropping2D(
                cropping=((abs(padding_left), abs(padding_right)), (abs(padding_left), abs(padding_right))))
        else:
            # This happens with the 3x3 stride 2 convs
            # This happens with the 7x7 stride 2 conv
            # This happens with the 3x3 stride 2 maxpool
            padder = ZeroPadding2D(padding=((padding_left, padding_right), (padding_left, padding_right)))
        return sequential(padder, operation)(tensor)

    return build


def conv2d_like_pytorch(filters, kernel_size, strides=1, padding='same', use_bias=False, **kwargs):
    """
    When 'same' padding a 2-stride convolution for an even input, there is an ambiguous situation
    and Pytorch and Tensorflow handle things differently.

    Example:
        - Input size 224, kernel_width 7, stride 2, padding 'same'
        - Desired output size is 112
        - Goal is to pad the input and then do a 'valid' convolution
        - Need 1 valid location for the first operation, and then 2 * 111 more, one for each stride
        - Therefore need 2 * 111 + 1 = 223 valid locations
        - Each valid location needs a buffer of kernel_size // 2 = 3 on each side
        - Therefore need a width of 223 + 2 * 3 = 229
        - Therefore need padding of 229 - 224 = 5
        - Here is the ambiguity. Is that 3 on the left and 2 on the right, or 2 on the left and 3 on the right?
        - For an input of 223 this would have been a clean 2 and 2

    Solution: Rather than figure out a universal solution, I will simply catch the three cases that come up on the basic
    224 resnet. i.e. the 7x7 stride 2 conv on 224 at the start, and the various 3x3 stride 2 convs on even sizes, and
    the 1x1 stride 2 convs on even sizes throughout the network.
    """

    # Innocent 3x3 stride 1 conv, no change
    if strides == 1:
        return Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, **kwargs)
    elif strides == 2 and padding == 'same':
        return same_pad_like_pytorch(
            Conv2D(filters, kernel_size, strides=strides, padding='valid', use_bias=use_bias, **kwargs),
            kernel_size
        )


def max_pooling_2d_like_pytorch(pool_size, strides=2, padding='same', **kwargs):
    # Exact same issue as conv above
    if strides == 2 and padding == 'same':
        return same_pad_like_pytorch(
            MaxPooling2D(pool_size, strides=strides, padding='valid', **kwargs),
            pool_size
        )


def batch_norm_2d_like_pytorch(**kwargs):
    return BatchNormalization(momentum=0.1, epsilon=1e-5, **kwargs)


def conv3x3(out_planes, stride=1):
    return conv2d_like_pytorch(out_planes, 3, strides=stride, padding='same', use_bias=False)


class BasicBlock:
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None):
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = batch_norm_2d_like_pytorch()
        self.relu1 = Activation('relu')
        self.conv2 = conv3x3(planes)
        self.bn2 = batch_norm_2d_like_pytorch()
        self.downsample = downsample
        self.relu2 = Activation('relu')
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = add([out, residual])
        out = self.relu2(out)

        return out


class Bottleneck:
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None):
        self.conv1 = conv2d_like_pytorch(planes, 1, use_bias=False)
        self.bn1 = batch_norm_2d_like_pytorch()
        self.relu1 = Activation('relu')
        self.conv2 = conv2d_like_pytorch(planes, 3, strides=stride, use_bias=False)
        self.bn2 = batch_norm_2d_like_pytorch()
        self.relu2 = Activation('relu')
        self.conv3 = conv2d_like_pytorch(planes * 4, 1, use_bias=False)
        self.bn3 = batch_norm_2d_like_pytorch()
        self.relu3 = Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = add([out, residual])
        out = self.relu3(out)

        return out


class ResNet:
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        self.conv1 = conv2d_like_pytorch(64, 7, strides=2, padding='same', use_bias=False, name='conv1')
        self.bn1 = batch_norm_2d_like_pytorch(name='bn1')
        self.relu = Activation('relu')
        self.maxpool = max_pooling_2d_like_pytorch(3, strides=2, padding='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AveragePooling2D(7, strides=1)
        self.fc = Dense(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = sequential(
                conv2d_like_pytorch(planes * block.expansion, 1, strides=stride, use_bias=False),
                batch_norm_2d_like_pytorch()
            )

        layers = []
        layers.append(block(planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes))

        return sequential(*layers)

    def build(self, input_shape):
        input = Input(shape=input_shape)

        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = Flatten()(x)
        x = self.fc(x)

        return Model(input, x)


def resnet18(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model.build((224, 224, 3))


def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model.build((224, 224, 3))


def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model.build((224, 224, 3))


def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model.build((224, 224, 3))


def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model.build((224, 224, 3))
