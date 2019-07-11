import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn import datasets, model_selection
import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"

from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.datasets import fetch_mldata
import pandas as pd
from sklearn.datasets import fetch_mldata
from sklearn.datasets import fetch_openml

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

mnist = fetch_openml("mnist_784",version=1)

mnist_data = mnist.data / 255

pd.DataFrame(mnist_data)
print(pd.DataFrame(mnist_data))

plt.imshow(mnist_data[10].reshape(28,28),cmap=cm.gray_r)
plt.show()