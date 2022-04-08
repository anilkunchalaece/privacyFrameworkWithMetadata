import torch
from torchvision import models
import torch.nn as nn


class AttrNet(nn.Module):
    
    def __init__(self,n_attr):
        super(AttrNet,self).__init__()

        self.net = models.resnet50(pretrained=True)

        # freeze layers of pre-trained model
        for param in self.net.parameters():
            param.requires_grad = False
        
        fc_inp = self.net.fc.in_features # take a note of no of i/p variables for last fc layer
        self.net.fc = nn.Sequential(
            nn.Linear(fc_inp,n_attr)
        )
        
    def forward(self,x):
        return self.net(x)



if __name__ == "__main__" :
    attrN = AttrNet(35)
    print(attrN.net)
