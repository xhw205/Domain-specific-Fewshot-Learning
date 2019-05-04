import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNEncoder(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=0),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
                        nn.BatchNorm2d(128),
                        nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 10),
        )
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        # out = self.fc(out)
        out = F.adaptive_avg_pool2d(out,1).squeeze(2).squeeze(2)
        return out #0.6*F.adaptive_avg_pool2d(out,1).squeeze(2).squeeze(2)+0.4*F.adaptive_max_pool2d(out,1).squeeze(2).squeeze(2)