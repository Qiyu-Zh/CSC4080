import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class Confusion_model(nn.Module):
    def __init__(self):
        '''
        Define the layers that you need
        '''
        super(Confusion_model, self).__init__()
        ### Begin your code ###
        self.model1=models.resnet18(pretrained=True)
        self.model1.fc=nn.Sequential(
            nn.Linear(self.model1.fc.in_features, 8), 
            nn.Tanh(),
            nn.Linear(8, 3)
        )

        self.model2=models.alexnet(pretrained=True)
        self.model2.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 3),
        )

        self.model3=timm.create_model('convnext_tiny',pretrained=True)
        self.model3.head.fc=nn.Sequential(
            nn.Linear(self.model3.head.fc.in_features, 8), 
            nn.Tanh(),
            nn.Linear(8, 3)
        )
        self.w=nn.Parameter(torch.ones(3))
        
    def forward(self, x):
        '''
        Define the forward propagation for data sample x
        '''
        ### Begin your code ###
        p1=self.model1(x)
        p2=self.model2(x)
        p3=self.model3(x)
        return self.w[0]*p1+self.w[1]*p2+self.w[2]*p3
        ### End your code ###






