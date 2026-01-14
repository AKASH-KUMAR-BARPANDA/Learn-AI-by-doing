import numpy as np

# f(x) = Y = Mx + C  -- > linear regression --> straight line equation
# f(x) = Y = (weight)x + Bias

# here --> F(x) = 2 * x , here 2 is to predict
X = np.array([1,2,3,4,5],dtype=np.float32)
Y = np.array([2,4,6,8,10],dtype=np.float32)

# beginning
weight = 0.00

# model prediction
def forward(x):
    return weight * x

# loss => MSE(mean square error) in case of linear regression
def loss(y,y_predicted):
    return ((y_predicted - y)**2).mean()

# gradient -> tells the model how much and in what direction
#               to adjust its parameters (weights/biases) to decrease the error

# MSE => 1/N  * (w*x - y)**2
# dJ/dw => 1/N 2*X (w*x -y)  ,, where w*x => Y = f(X)

def gradient(x,y,y_predicted):
    return np.dot(2*x,y_predicted - y).mean()


value = 5
print(f"prediction before training f({value}) = {forward(value):.3f}")

# training
learning_rate = 0.01 # , jump to neighbor to satisfy
n_iter = 20


for epoch in range(n_iter):
    #prediction ==> forward pass
    y_predicted = forward(X)

    # loss
    l = loss(Y,y_predicted)

    # gradient
    dw = gradient(X,Y,y_predicted)

    #update weight
    weight -= learning_rate * dw

    if epoch % 2 == 0:
        print(f"epoch {epoch + 1 }: weight = {weight:.3f} , loss = {l:.8f} ")

print(f"prediction After training f(5) = {forward(value):.3f}")






