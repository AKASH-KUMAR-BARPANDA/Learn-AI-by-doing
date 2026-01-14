import torch

x = torch.rand(5,3)
y = torch.rand(5,3)
# x+y ,, torch.add(x,y) # -> inplace x.add_(y)

# x[:,:]  --> first is row and second is column
# x[].item() --> actual value od the single item

#.view

