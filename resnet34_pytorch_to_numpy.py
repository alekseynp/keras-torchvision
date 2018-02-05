import torchvision
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable

# Load the model
model = torchvision.models.resnet34(pretrained=True)

# Write state_dict to numpy arrays on disk
sd = model.state_dict()
for k, v in sd.items():
    np.save(k, v.numpy())

# Process a test image for comparison to Keras results
img = Image.open('test_image.png')
img = img.resize((224,224))
img_array = np.array(img)
img_array = img_array / 255.
img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224,  0.225])
img_array = np.transpose(img_array, (2,0,1))[None, :, :, :]

input = torch.FloatTensor(img_array)
model.eval()
result = model(Variable(input))
np.save('pytorch_output_resnet34.npy', result.data.numpy())