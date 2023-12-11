import torch
import torch.nn as nn
import torch.nn.functional as f

class MyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        '''
        Initialize some essential hyperparameters for your loss function
        '''
        super(MyLoss, self).__init__()
        ### Begin your code ###
        self.gamma = gamma
        self.alpha=alpha
        ### End your code ###

    def forward(self, outputs, labels):
        '''
        Define the calculation of the loss
        '''
        ### Begin your code ###
     
        logpt = -f.cross_entropy(outputs,labels)
 
    
        loss = - (1 - torch.exp(logpt)) ** self.gamma * logpt
 
        return loss
        ### End your code ###