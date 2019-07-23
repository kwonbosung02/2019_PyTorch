import urllib

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_digits
from sklearn import datasets, model_selection

from matplotlib import pyplot as plt
from matplotlib import cm

import pandas as pd

from scipy.io import loadmat
#mnist_alternative_url = "https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat"
mnist_path = "./mnist-original.mat"

#response = urllib.request.urlopen(mnist_alternative_url)

#with open(mnist_path, "wb") as f:
 #   content = response.read()
  #  f.write(content)

mnist_raw = loadmat(mnist_path)
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
}

print("Success")

mnist_data = mnist['data'] / 255

pd.DataFrame(mnist_data)
mnist_label = mnist['target']
mnist_label

train_size = 5000
test_size = 500
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data,mnist_label,train_size=train_size,test_size=test_size)
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).long()


test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).long()

print(train_X.shape)
print(train_Y.shape)

train = TensorDataset(train_X, train_Y)


train_loader = DataLoader(train, batch_size=100, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784,256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,128)
        self.fc6 = nn.Linear(128,10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.dropout(x, training=self.training)
        x = self.fc6(x)
        return F.log_softmax(x)


model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(1000):
    total_loss = 0

    for train_x, train_y in train_loader:

        train_x, train_y = Variable(train_x), Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.data

    print(epoch+1,total_loss)

test_x, test_y = Variable(test_X), Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.cpu().data.numpy() == result.cpu().numpy()) / len(test_y.cpu().data.numpy())

print(accuracy)