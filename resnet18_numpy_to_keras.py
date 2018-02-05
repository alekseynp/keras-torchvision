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


def process_layer(layer_type, keras_layer_name, torchvision_layer_name):
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


model = torchvision_resnet.resnet18()

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

conversion_tasks = [
    ('conv2d', 'conv1', 'conv1'), ('batchnorm', 'bn1', 'bn1'),
    ('conv2d', 'conv2d_1', 'layer1.0.conv1'), ('batchnorm', 'batch_normalization_1', 'layer1.0.bn1'),
    ('conv2d', 'conv2d_2', 'layer1.0.conv2'), ('batchnorm', 'batch_normalization_2', 'layer1.0.bn2'),
    ('conv2d', 'conv2d_3', 'layer1.1.conv1'), ('batchnorm', 'batch_normalization_3', 'layer1.1.bn1'),
    ('conv2d', 'conv2d_4', 'layer1.1.conv2'), ('batchnorm', 'batch_normalization_4', 'layer1.1.bn2'),
    ('conv2d', 'conv2d_5', 'layer2.0.downsample.0'), ('batchnorm', 'batch_normalization_5', 'layer2.0.downsample.1'),
    ('conv2d', 'conv2d_6', 'layer2.0.conv1'), ('batchnorm', 'batch_normalization_6', 'layer2.0.bn1'),
    ('conv2d', 'conv2d_7', 'layer2.0.conv2'), ('batchnorm', 'batch_normalization_7', 'layer2.0.bn2'),
    ('conv2d', 'conv2d_8', 'layer2.1.conv1'), ('batchnorm', 'batch_normalization_8', 'layer2.1.bn1'),
    ('conv2d', 'conv2d_9', 'layer2.1.conv2'), ('batchnorm', 'batch_normalization_9', 'layer2.1.bn2'),
    ('conv2d', 'conv2d_10', 'layer3.0.downsample.0'), ('batchnorm', 'batch_normalization_10', 'layer3.0.downsample.1'),
    ('conv2d', 'conv2d_11', 'layer3.0.conv1'), ('batchnorm', 'batch_normalization_11', 'layer3.0.bn1'),
    ('conv2d', 'conv2d_12', 'layer3.0.conv2'), ('batchnorm', 'batch_normalization_12', 'layer3.0.bn2'),
    ('conv2d', 'conv2d_13', 'layer3.1.conv1'), ('batchnorm', 'batch_normalization_13', 'layer3.1.bn1'),
    ('conv2d', 'conv2d_14', 'layer3.1.conv2'), ('batchnorm', 'batch_normalization_14', 'layer3.1.bn2'),
    ('conv2d', 'conv2d_15', 'layer4.0.downsample.0'), ('batchnorm', 'batch_normalization_15', 'layer4.0.downsample.1'),
    ('conv2d', 'conv2d_16', 'layer4.0.conv1'), ('batchnorm', 'batch_normalization_16', 'layer4.0.bn1'),
    ('conv2d', 'conv2d_17', 'layer4.0.conv2'), ('batchnorm', 'batch_normalization_17', 'layer4.0.bn2'),
    ('conv2d', 'conv2d_18', 'layer4.1.conv1'), ('batchnorm', 'batch_normalization_18', 'layer4.1.bn1'),
    ('conv2d', 'conv2d_19', 'layer4.1.conv2'), ('batchnorm', 'batch_normalization_19', 'layer4.1.bn2'),
    ('dense', 'dense_1', 'fc')
]

for t in conversion_tasks:
    process_layer(*t)

img = Image.open('test_image.png')
img = img.resize((224, 224))
img_array = np.array(img)
img_array = img_array / 255.
img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

predictions = model.predict(img_array[None, :])

predictions_torchvision = np.load('pytorch_output_resnet18.npy')

plt.figure()
plt.scatter(predictions.flatten(), predictions_torchvision.flatten())
plt.xlabel('Keras Logit Value')
plt.ylabel('Pyotrch Logit Value')
plt.title('Resnet18')
plt.savefig('resnet18.png')

model.save('resnet18.h5')