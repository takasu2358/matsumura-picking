# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#モデルの作成
class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        # the size of the inputs to each layer will be inferred
        self.conv1a=nn.Conv2d(1,32,16)
        self.conv2a=nn.Conv2d(32,64,8)
        self.conv1b=nn.Conv2d(1,32,16)
        self.conv2b=nn.Conv2d(32,64,8)
        self.conv3=nn.Conv2d(64,64,5)
        self.conv4=nn.Conv2d(64,64,3)
        self.l1=nn.Linear(1024,1024)
        self.l2=nn.Linear(1024,1024)
        self.l3=nn.Linear(1024,2)
        self.dropout = nn.Dropout(0.5)

model = CNN1()

found = "./Method1/SuccessFailure_Work4"
Path = found + ".model13"
xy = np.load(Path)

model.conv1a.weight = nn.Parameter(torch.tensor(xy["conv1a/W"]))
model.conv1a.bias = nn.Parameter(torch.tensor(xy["conv1a/b"]))
model.conv2a.weight = nn.Parameter(torch.tensor(xy["conv2a/W"]))
model.conv2a.bias = nn.Parameter(torch.tensor(xy["conv2a/b"]))
model.conv1b.weight = nn.Parameter(torch.tensor(xy["conv1b/W"]))
model.conv1b.bias = nn.Parameter(torch.tensor(xy["conv1b/b"]))
model.conv2b.weight = nn.Parameter(torch.tensor(xy["conv2b/W"]))
model.conv2b.bias = nn.Parameter(torch.tensor(xy["conv2b/b"]))
model.conv3.weight = nn.Parameter(torch.tensor(xy["conv3/W"]))
model.conv3.bias = nn.Parameter(torch.tensor(xy["conv3/b"]))
model.conv4.weight = nn.Parameter(torch.tensor(xy["conv4/W"]))
model.conv4.bias = nn.Parameter(torch.tensor(xy["conv4/b"]))
model.l1.weight = nn.Parameter(torch.tensor(xy["l1/W"]))
model.l1.bias = nn.Parameter(torch.tensor(xy["l1/b"]))
model.l2.weight = nn.Parameter(torch.tensor(xy["l2/W"]))
model.l2.bias = nn.Parameter(torch.tensor(xy["l2/b"]))
model.l3.weight = nn.Parameter(torch.tensor(xy["l3/W"]))
model.l3.bias = nn.Parameter(torch.tensor(xy["l3/b"]))

pathname = found + ".pth"
torch.save(model.state_dict(), pathname)
