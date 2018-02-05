# keras-torchvision

Convert the pretrained Resnet18 from Pytorch's torchvision module for use in Keras with the Tensorflow backend.

Divided into two scripts, using numpy files as an interchange medium. This makes it possible to have Pytorch and Tensorflow in two different repositories.

Converts with anal attention to detail, matching activations at each layer. Picky about how 'same' padding is implemented.

`torchvision_resnet.py` implemented to read very similarly to `pytorch/vision/torchvision/models/resnet.py`

Usage:

```
python resnet18_pytorch_to_numpy.py
python resnet18_numpy_to_keras.py

python
import keras
keras.load_model('resnet18.h5')
```

Validation:

![Test Image](test_image.png)

![Resnet18](resnet18.png)

Tested:
- Kears 2.1.2
- Tensorflow 1.4.1
- Pytorch 0.2.0_3
- Cuda 7.5
