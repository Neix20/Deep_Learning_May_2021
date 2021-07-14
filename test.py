import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.utils.data import DataLoader

from torchsummary import summary
 
trainset =  torchvision.datasets.FashionMNIST(root=".", train= True,download= True)
testset  = torchvision.datasets.FashionMNIST(root=".", train=False, download=True)

print(trainset.classes)