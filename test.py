import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.utils.data import DataLoader

from torchsummary import summary
 
trainset =  torchvision.datasets.FashionMNIST(root=".",transform=None, train= True,download= True)
testset  = torchvision.datasets.FashionMNIST(root=".", transform=None,train=False, download=True)


img, label = trainset.data[3], trainset.targets[3]

print(img.dtype)
print(img.shape)

plt.title(trainset.classes[label])
plt.imshow(img)
_ = plt.axis("off")
plt.show()