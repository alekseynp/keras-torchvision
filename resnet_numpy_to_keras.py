import argparse
import numpy as np
import torchvision_resnet
import matplotlib.pyplot as plt
from PIL import Image
import os


def convert_kernel(x):
    return np.transpose(x, (2, 3, 1, 0))


def convert_dense(x):
    return np.transpose(x)


def parse_keras_name(x):
    return x.split('/')[1][:-2]


def process_layer(model, layer_type, keras_layer_name, torchvision_layer_name):
    layer = model.get_layer(keras_layer_name)
    weight_names = [x.name for x in layer.weights]
    weight_names = [parse_keras_name(x) for x in weight_names]
    tv_weight_names = [weight_name_mapping[layer_type][x] for x in weight_names]
    tv_filenames = [torchvision_layer_name + '.' + x + '.npy' for x in tv_weight_names]
    tv_weights = [np.load(x) for x in tv_filenames]
    tv_weights = [conversion_functions[layer_type](x) for x in tv_weights]
    layer.set_weights(tv_weights)
    for f in tv_filenames:
        os.remove(f)


def get_conversion_tasks(config, is_basic):
    conversion_tasks = [('conv2d', 'conv1', 'conv1'), ('batchnorm', 'bn1', 'bn1'), ]

    conv_counter = 1
    bn_counter = 1
    for layer in range(4):
        if not is_basic or layer > 0:
            # 0th layer of basic doesn't have a downsample
            conversion_tasks.append(
                ('conv2d',
                 'conv2d_{}'.format(conv_counter),
                 'layer{}.0.downsample.0'.format(layer + 1)))
            conv_counter += 1
            conversion_tasks.append(
                ('batchnorm',
                 'batch_normalization_{}'.format(bn_counter),
                 'layer{}.0.downsample.1'.format(layer + 1)))
            bn_counter += 1
        for block in range(config[layer]):
            for conv_bn in range(2 if is_basic else 3):
                conversion_tasks.append(('conv2d', 'conv2d_{}'.format(conv_counter),
                                         'layer{}.{}.conv{}'.format(layer + 1, block, conv_bn + 1)))
                conv_counter += 1
                conversion_tasks.append(('batchnorm', 'batch_normalization_{}'.format(bn_counter),
                                         'layer{}.{}.bn{}'.format(layer + 1, block, conv_bn + 1)))
                bn_counter += 1

    conversion_tasks.append(('dense', 'dense_1', 'fc'))

    return conversion_tasks

weight_name_mapping = {
    'dense': {
        'kernel': 'weight',
        'bias': 'bias'
    },
    'conv2d': {
        'kernel': 'weight'
    },
    'batchnorm': {
        'gamma': 'weight',
        'beta': 'bias',
        'moving_mean':  'running_mean',
        'moving_variance': 'running_var'
    }
}

conversion_functions = {
    'conv2d': convert_kernel,
    'batchnorm': lambda x: x,
    'dense': convert_dense
}

BOTTLENECK_FAMILY = {
    '50': [3, 4, 6, 3],
    '101': [3, 4, 23, 3],
    '152': [3, 8, 36, 3]
}

BASIC_FAMILY = {
    '18': [2, 2, 2, 2],
    '34': [3, 4, 6, 3]
}

MODEL_BUILDERS = {
    '18': torchvision_resnet.resnet18,
    '34': torchvision_resnet.resnet34,
    '50': torchvision_resnet.resnet50,
    '101': torchvision_resnet.resnet101,
    '152': torchvision_resnet.resnet152
}

def convert(version):
    model = MODEL_BUILDERS[version]()

    if version in BOTTLENECK_FAMILY:
        conversion_tasks = get_conversion_tasks(BOTTLENECK_FAMILY[version], is_basic=False)
    elif version in BASIC_FAMILY:
        conversion_tasks = get_conversion_tasks(BASIC_FAMILY[version], is_basic=True)

    for t in conversion_tasks:
        process_layer(model, *t)

    img = Image.open('test_image.png')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.
    img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

    predictions = model.predict(img_array[None, :])

    predictions_torchvision = np.load('pytorch_output_resnet{}.npy'.format(version))

    plt.figure()
    plt.scatter(predictions.flatten(), predictions_torchvision.flatten())
    plt.xlabel('Keras Logit Value')
    plt.ylabel('Pyotrch Logit Value')
    plt.title('Resnet{}'.format(version))
    plt.savefig('resnet{}.png'.format(version))

    model.save('resnet{}.h5'.format(version))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', choices=['18', '34', '50', '101', '152'], help='Which resnet to convert.')
    args = parser.parse_args()
    convert(args.version)