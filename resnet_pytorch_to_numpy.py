import argparse
import torchvision
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable


MODEL_BUILDERS = {
    '18': torchvision.models.resnet18,
    '34': torchvision.models.resnet34,
    '50': torchvision.models.resnet50,
    '101': torchvision.models.resnet101,
    '152': torchvision.models.resnet152
}


def convert(version):
    # Load the model
    model = MODEL_BUILDERS[version](pretrained=True)

    # Write state_dict to numpy arrays on disk
    sd = model.state_dict()
    for k, v in sd.items():
        np.save(k, v.numpy())

    # Process a test image for comparison to Keras results
    img = Image.open('test_image.png')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.
    img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img_array = np.transpose(img_array, (2, 0, 1))[None, :, :, :]

    input = torch.FloatTensor(img_array)
    model.eval()
    result = model(Variable(input))
    np.save('pytorch_output_resnet{}.npy'.format(version), result.data.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('version', choices=['18', '34', '50', '101', '152'], help='Which resnet to convert.')
    args = parser.parse_args()
    convert(args.version)
