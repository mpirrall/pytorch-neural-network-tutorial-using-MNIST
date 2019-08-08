import torch.nn as nn

class NeuralNet(nn.Module):

    def __init__(self):
        super(NeuralNet, self).__init__()
        #For the first layer, the input is the 784 pixels of the 28x28 image, and output is the data from its 100 nodes
        self.linear1 = nn.Linear(784, 100)
        #The second layer takes as input the data from the nodes in the first layer, and outputs the data of its 50 nodes
        self.linear2 = nn.Linear(100, 50)
        #The third layer takes as input the data from the nodes in the second later, and outputs the confidence
        #for each of the digits (0-9).  Each of the 10 outputs corresponds to its respective digit
        self.linear3 = nn.Linear(50, 10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        x = self.linear3(x)
        return x