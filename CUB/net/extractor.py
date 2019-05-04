import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
class ClassificationNetwork(nn.Module):
    def __init__(self):
        super(ClassificationNetwork, self).__init__()
        self.convnet = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.convnet.fc.in_features
        self.convnet.fc = nn.Linear(num_ftrs,100)

    def forward(self,inputs):
        outputs = self.convnet(inputs)
        return outputs

class EmbeddingNetwork(nn.Module):  #OrderNet
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.resnet = ClassificationNetwork()
        self.conv1 = self.resnet.convnet.conv1
        self.conv1.load_state_dict(self.resnet.convnet.conv1.state_dict())
        self.bn1 = self.resnet.convnet.bn1
        self.bn1.load_state_dict(self.resnet.convnet.bn1.state_dict())
        self.relu = self.resnet.convnet.relu
        self.maxpool = self.resnet.convnet.maxpool
        self.layer1 = self.resnet.convnet.layer1
        self.layer1.load_state_dict(self.resnet.convnet.layer1.state_dict())
        self.layer2 = self.resnet.convnet.layer2
        self.layer2.load_state_dict(self.resnet.convnet.layer2.state_dict())
        self.layer3 = self.resnet.convnet.layer3
        self.layer3.load_state_dict(self.resnet.convnet.layer3.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.layer4 = self.resnet.convnet.layer4
        self.layer4.load_state_dict(self.resnet.convnet.layer4.state_dict())
        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.avgpool = self.resnet.convnet.avgpool
        self.fc1 = nn.Linear(512*1, 2)
        self.fc2 = nn.Linear(512*1, 100)

    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        out = layer4
        x= self.avgpool(out)
        x = x.view(x.size(0), -1)
        ip1 = self.fc1(x)
        ip2 = self.fc2(x)
        return x, ip1, ip2