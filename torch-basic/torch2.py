
### AUTOGRAD

import torch

x = torch.rand(3,requires_grad=True)
print(x)

y = x+2
print(y)

z = y*y*2
#z = z.mean()
print(z)

v = torch.tensor([0.1,0.1,0.001],dtype=torch.float32)
z.backward(v)    # not problem because I used mean, But in real I need to multiplay vector
print(x.grad)
