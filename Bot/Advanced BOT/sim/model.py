import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet,self).__init__()

        self.l1 = nn.Linear(in_features=input_size,out_features = hidden_size)
        self.l2 = nn.Linear(in_features=hidden_size,out_features = hidden_size)
        self.l3 = nn.Linear(in_features=hidden_size,out_features = output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self,x):

        # fc1
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)

        #fc2
        x = self.l2(x)
        x = self.relu(x)
        x = self.dropout(x)

        #fc3
        x = self.l3(x)
        # no activation function or softmax required !!

        return x

