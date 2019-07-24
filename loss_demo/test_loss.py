import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from CenterLoss import CenterLoss
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import csv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.prelu1_2 = nn.PReLU()
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.prelu2_2 = nn.PReLU()
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.prelu3_2 = nn.PReLU()
        self.preluip1 = nn.PReLU()
        self.ip1 = nn.Linear(128*3*3, 2)
        self.ip2 = nn.Linear(128*3*3, 10, bias=False)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x,2)
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1, 128*3*3)
        ip2 = self.ip2(x)
        ip1 = self.preluip1(self.ip1(x))
        return ip1, F.log_softmax(ip2, dim=1)

def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)
def supp_idxs(t, c):
    return t.eq(c).nonzero().squeeze(1)
num = 0
max_ = 0.0
save = []
def train(num, epoch):
    running_loss = 0.0
    batchtotal = 0.0
    ip1_loader = []
    idx_loader = []
    totalloss = 0
    for i,(data, targets) in enumerate(train_loader):
        batchtotal+=data.size(0)
        data, targets = data.to(device), targets.to(device)
        ip1, pred = model(data)
        c, centers = centerloss(targets, ip1)
        loss = nllloss(pred, targets) + loss_weight * c#centerloss
        cls = torch.unique(targets.cpu())
        s_idx = [supp_idxs(targets.cpu(), i) for i in cls]  # idx
        pre = torch.stack([ip1[idx_list].mean(0) for idx_list in s_idx])
        dists = euclidean_dist(pre, centers)
        log_p_y = F.log_softmax(-dists, dim=1).view(len(s_idx), 10)
        target_inds = cls.view(cls.size(0), -1).cuda()
        _, y_hat = torch.max(log_p_y.data, 1)
        loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()
        loss = loss + 0.6*loss_val  #CN_loss 
        optimizer4nn.zero_grad()
        optimzer4center.zero_grad()
        loss.backward()
        optimizer4nn.step()
        optimzer4center.step()
    return num+1

def _testfashion(epoch):
    model.eval()
    batchtotal = 0
    correct = 0.0
    for i,(data, targets) in enumerate(test_loader):
        batchtotal+=data.size(0)
        data, targets = data.to(device), targets.to(device)
        ip1, outputs = model(data)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum((preds) == targets.data)
    print('Test===>{}'.format(correct.double()/batchtotal))


use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# Dataset
trainset = datasets.FashionMNIST('./fashionMNIST', download=True,train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
testset = datasets.FashionMNIST('./fashionMNIST', download=True, train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = DataLoader(testset, batch_size=128, shuffle=False)
# Model
model = Net().to(device)
# NLLLoss
nllloss = nn.NLLLoss().to(device) #CrossEntropyLoss = log_softmax + NLLLoss
# CenterLoss
loss_weight = 1
centerloss = CenterLoss(10, 2).to(device)
# optimzer4nn
optimizer4nn = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer4nn,20,gamma=0.8)
# optimzer4center
optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)
total = 0.0
correct = 0.0
max_corr = 0.0
for epoch in range(100):
    sheduler.step()
    num = train(num,epoch+1)
    _testfashion(epoch)