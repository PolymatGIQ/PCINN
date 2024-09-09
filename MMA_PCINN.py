# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 16:08:19 2023

@author: Nick Ballard

Method for "Polymer Chemsitry Informed Neural Networks" that combines data driven method with Jacobian elements from pretrained first principles mathematical model
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.func import vmap, jacrev

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# %% Model structure
class NNmodel(nn.Module):  # Main network
    def __init__(self):
        super(NNmodel, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class DomainModel(nn.Module):  # "Theory network"
    def __init__(self):
        super(DomainModel, self).__init__()
        self.fc1 = nn.Linear(5, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.Xout = nn.Linear(64, 1)
        self.Mout = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        X_out = F.sigmoid(self.Xout(x))  # conversion
        M_out = F.softplus(self.Mout(x))  # Mn, Mw, Mz, Mz+1, Mv

        return torch.cat((X_out, M_out), dim=-1)


# %% Scale features to between 0 and 1
def scalefeaturezeroone(x, scalerxmax, scalerxmin):
    max_minus_min = (scalerxmax - scalerxmin)
    return (x - scalerxmin) / max_minus_min


# %% Get all the data and load models
scalerx_max = np.load('scalerx_max.npy')
scalerx_min = np.load('scalerx_min.npy')

# load training data
df = pd.read_excel('PMMAordered.xlsx')

dfX = df[["[M]", "[S]", "[I]", "temp", "time", "Reaction"]]
dfY = df[["X", "Mn", "Mw", "Mz", "Mzplus1", "Mv"]]

Xdata = dfX.values
Ydata = dfY.values

Ydata[:, 1:] = np.log10(Ydata[:, 1:])
Xdata[:, :5] = scalefeaturezeroone(Xdata[:, :5], scalerx_max, scalerx_min)

dfGPC = df.iloc[:, 18:]
GPCdata = dfGPC.values

Domain_NN = DomainModel()
Domain_NN.load_state_dict(torch.load('MMA_solution_net.pt'))  # loads pretrained "theory" model

# %% Define the region over which Jacobian is sampled
Tupper = 273 + 90
Tlower = 273 + 50

Mupper = 5
Mlower = 0.5

Iupper = 0.1
Ilower = 0.005

timeupper = 10 * 60 * 60
timelower = 5 * 60

M_sampler = torch.distributions.Uniform(low=Mlower, high=Mupper)
T_sampler = torch.distributions.Uniform(low=Tlower, high=Tupper)
I_sampler = torch.distributions.Uniform(low=Ilower, high=Iupper)
time_sampler = torch.distributions.Uniform(low=timelower, high=timeupper)

# %% Run training loop with leave one out cross validation
TestReaction = 8  # Integer between 1 and 8. Picks reaction that will be tested on (other reactions for training)

# get training samples
Xtrainsample = Xdata[Xdata[:, 5] != TestReaction]
Xtrainsample = Xtrainsample[:, :5]
Xtrainsample = torch.from_numpy(Xtrainsample).float()
Ytrainsample = Ydata[Xdata[:, 5] != TestReaction]
Ytrainsample = torch.from_numpy(Ytrainsample).float()

# get test samples
Xtestsample = Xdata[Xdata[:, 5] == TestReaction]
Xtestsample = Xtestsample[:, :5]
Xtestsample = torch.from_numpy(Xtestsample).float()
Ytestsample = Ydata[Xdata[:, 5] == TestReaction]

# %% Train normal Neural network
model = NNmodel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3E-4)

epochs = 10000
reg_losses = []
reg_losses_test = []
for epoch in range(epochs):
    Sum_Obj_loss = 0
    pred = model(Xtrainsample)
    trainloss = loss_function(pred, Ytrainsample)
    Sum_Obj_loss += trainloss
    if epoch >= 1:
        # Backpropagation
        trainloss.backward()
        optimizer.step()
        optimizer.zero_grad()

    pred = model(Xtestsample)
    testloss = loss_function(pred, torch.from_numpy(Ytestsample).float())
    reg_losses.append(float(Sum_Obj_loss))
    reg_losses_test.append(float(testloss))

NNpred = model(Xtestsample)

plt.plot(np.log(reg_losses), label='Training loss')
plt.plot(np.log(reg_losses_test), label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# %% Train PCINN
PCINNmodel = NNmodel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(PCINNmodel.parameters(), lr=3E-4)

epochs = 10000

reg_losses = []
reg_losses_test = []

totaljacsamples = 32

for epoch in range(epochs):
    Sum_Jac_loss = 0
    Sum_Obj_loss = 0

    pred = PCINNmodel(Xtrainsample)
    trainloss = loss_function(pred, Ytrainsample)
    Sum_Obj_loss += trainloss

    Msample = M_sampler.sample((32, 1))
    Ssample = 10 - Msample
    Isample = I_sampler.sample((32, 1))
    Tsample = T_sampler.sample((32, 1))
    tsample = time_sampler.sample((32, 1))
    sampl = torch.cat((Msample, Ssample, Isample, Tsample, tsample), 1)
    sampl = (sampl - scalerx_min) / (scalerx_max - scalerx_min)

    jac_theory_sampl = vmap(jacrev(Domain_NN))(sampl.float())
    jac_sampl = vmap(jacrev(PCINNmodel))(sampl.float())

    jacloss = loss_function(jac_sampl, jac_theory_sampl)
    Sum_Jac_loss += jacloss

    loss = trainloss + jacloss

    if epoch >= 1:
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    reg_losses.append(float(Sum_Obj_loss))
    pred = PCINNmodel(Xtestsample)
    testloss = loss_function(pred, torch.from_numpy(Ytestsample).float())
    reg_losses_test.append(float(testloss))

EBNNpred = PCINNmodel(Xtestsample)

plt.plot(np.log(reg_losses), label='Training loss')
plt.plot(np.log(reg_losses_test), label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
