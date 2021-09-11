import numpy as np
import matplotlib.pyplot as plot
import matplotlib as mpl
import torchvision.transforms
from PIL import Image
import torch

image = Image.open("car.jpeg")
loader = torchvision.transforms.ToTensor()
unloader = torchvision.transforms.ToPILImage()
imageTensor = loader(pic=image)
tensor = imageTensor.unsqueeze(0)
print(tensor)
filt = np.array([[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]])
conv = torch.nn.Conv2d(3, 1, (3, 3))
x = conv(tensor)
img = x.squeeze(0)
newImage = unloader(img)
newImage.show()
