import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.datasets.mnist import FashionMNIST
import torchvision.transforms as transforms

import torch.optim as optim
from torch.utils.data import DataLoader

from torchsummary import summary

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
])
trainset = FashionMNIST(root=".",transform=transform, train= True,download= True)
testset  = FashionMNIST(root=".",transform=transform,train= False, download=True)

img, label = trainset.data[3], trainset.targets[3]

print(trainset.classes)
plt.title(trainset.classes[label])
plt.imshow(img, cmap=plt.get_cmap('gray'))
_ = plt.axis("off")
plt.show()
'''
for i in range(9):
	# define subplot
	plt.subplot(330 + 1 + i)
	# plot raw pixel data
	plt.imshow(trainset.data[i], cmap=plt.get_cmap('gray'))
# show the figure
plt.show()
'''

print(trainset.classes)

def display(one_digit):
    one_digit_image = one_digit.reshape(28,28)
    plt.imshow (one_digit_image, cmap = matplotlib.cm.gray, interpolation = 'nearest')
    
one_digit = trainset.data[0]  # change number (0~69999) to show other images
display(one_digit)