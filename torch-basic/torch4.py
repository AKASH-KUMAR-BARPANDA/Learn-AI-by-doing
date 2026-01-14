import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8],[10]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_sample, n_feature = X.shape
input_size, output_size = n_feature, 1 # predicting 1 value

model = nn.Linear(input_size, output_size)

print(f"Prediction before training f(5) = {model(X_test).item():.3f}")

# Training Setup
learning_rate = 0.01
n_iter = 100 # Increased slightly for better convergence

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iter):
    # 1. Forward pass: Compute predicted y by passing x to the model
    y_predicted = model(X)

    # 2. Compute loss
    l = loss(y_predicted, Y)

    # 3. Backward pass
    l.backward()

    # 4. Update weights
    optimizer.step()

    # 5. Zero gradients
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: weight = {w[0][0].item():.3f}, loss = {l.item():.8f}")

print(f"Prediction after training f(5) = {model(X_test).item():.3f}")



#------------------ REVISED __________________________>>>>
# import torch
# import torch.nn as nn
#
# # 1. Reshaping to (Samples, Features) -> [[1],[2]...]
# X = torch.tensor([[1],[2],[3],[4],[5]], dtype=torch.float32)
# Y = torch.tensor([[2],[4],[6],[8],[10]], dtype=torch.float32)
#
# # input_size = 1 (we have 'x')
# # output_size = 1 (we want to predict 'y')
# model = nn.Linear(1, 1)
#
# learning_rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# loss_func = nn.MSELoss()
#
# for epoch in range(20):
#     # Forward pass: prediction = model(X)
#     y_pred = model(X)
#
#     # Loss: MSE
#     l = loss_func(y_pred, Y)
#
#     # Backward pass: This replaces your 'dw = gradient()' function
#     l.backward()
#
#     # Update: This replaces 'weight -= lr * dw'
#     optimizer.step()
#
#     # CRITICAL: PyTorch accumulates gradients. We must clear them for the next loop.
#     optimizer.zero_grad()