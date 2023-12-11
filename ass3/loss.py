import torch
import torch.nn as nn
import torch.nn.functional as f

class MyLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(MyLoss, self).__init__()
        self.gamma = gamma
        self.alpha=alpha
    def forward(self, pred, gt):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
     
        return f.binary_cross_entropy_with_logits(pred, gt,torch.tensor([30]).to(device))


