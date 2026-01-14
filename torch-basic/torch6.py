from symtable import Class

import torch
import torch.nn as nn
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



# preprocessing

bc  = datasets.load_breast_cancer()
X ,y = bc.data ,bc.target

n_samples, n_feature = X.shape
print(n_samples,n_feature)

X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.2,random_state=34)

#scale
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
Y_train = torch.from_numpy(Y_train.astype(np.float32))
Y_test = torch.from_numpy(Y_test.astype(np.float32))

Y_train = Y_train.view(Y_train.shape[0],1)
Y_test = Y_test.view(Y_test.shape[0],1)

# model
class Model(nn.Module):
    def __init__(self,input_size ,out_size):
        super(Model,self).__init__()

        self.L = nn.Linear(in_features=input_size,out_features= 1)

    def forward(self,x):
        y_predicted = self.L(x)
        y_predicted = torch.sigmoid(input=y_predicted)
        return y_predicted


model = Model(n_feature,1)



# loss and optimizer
losses = nn.BCELoss()
# optimizer --> model.parameter // remember
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# training loop

for epoch in range(1000):
    # forward
    Y_pred = model(X_train)
    #loss
    loss = losses(Y_pred,Y_train)

    # backward
    loss.backward()

    #optimizer
    optimizer.step()
    # zero optimizer grad
    optimizer.zero_grad()

    if epoch % 100 == 0 :
        print(f"epoch - {epoch} : loss = {loss.item():.4f}")
