import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# prepare data --> step 0
X_numpy, Y_numpy = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)


x = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(Y_numpy.astype(np.float32))


y = y.view(y.shape[0],1)
# feature -> the feature need to be taken from sample to train .
n_sample , n_feature = x.shape


# model --> step 1

model = nn.Linear(in_features=n_feature,out_features=1)


# loss and optimizer --> step 2
criterion = nn.MSELoss()
optimizer  = torch.optim.SGD(model.parameters(),lr=0.01)

# training loop --> step 3

for  epoch in range(1000):

    # forward pass
    Y_pred = model(x)

    # loss
    loss = criterion(Y_pred,y)
    # backward pass
    loss.backward()
    # optimizer
    optimizer.step()

    # zero grad
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f"epoch {epoch}  : loss = {loss.item():.4f}")

# plot
# detach tensor --> gradient
predicted = model(x).detach().numpy()

plt.plot(X_numpy,Y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.show()







