import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from  torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from CenterLoss import CenterLoss
import matplotlib.pyplot as plt
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
        self.ip2 = nn.Linear(2, 10, bias=False)

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

        ip1 = self.preluip1(self.ip1(x))
        ip2 = self.ip2(ip1)
        return ip1, F.log_softmax(ip2, dim=1)

def visualize(feat, labels, epoch):
    plt.ion()

    c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.clf()
    plt.tick_params(labelsize=14)

    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
    plt.xlim(xmin=-9,xmax=9)
    plt.ylim(ymin=-9,ymax=9)
    plt.text(-7.8,7.3,"epoch=%d" % epoch)
    plt.tick_params(labelsize=20)
    plt.savefig('./imgs/%d.jpg'%epoch )
    plt.draw()
    plt.pause(0.001)

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
    print("Training... Epoch = %d" % epoch)
    running_loss = 0.0
    batchtotal = 0
    ip1_loader = []
    idx_loader = []
    totalloss = 0
    for i,(data, targets) in enumerate(train_loader):
        batchtotal+=data.size(0)
        data, targets = data.to(device), targets.to(device)
        ip1, pred = model(data)
        c, centers = centerloss(targets, ip1)
        loss = nllloss(pred, targets) + loss_weight * c#centerloss(target, ip1)
        optimizer.zero_grad()
        optimzer4center.zero_grad()
        loss.backward()
        optimizer.step()
        optimzer4center.step()
        running_loss += loss.item() * data.size(0)
        ip1_loader.append(ip1)
        idx_loader.append((targets))
    epoch_loss = running_loss/ batchtotal
    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(),labels.data.cpu().numpy(),epoch)
    return num+1

use_cuda = torch.cuda.is_available() and True
device = torch.device("cuda" if use_cuda else "cpu")
# Dataset
trainset = datasets.FashionMNIST('./fashionMNIST', download=True,train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
# testset = datasets.FashionMNIST('./fashionMNIST', download=True, train=False, transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))]))
# test_loader = DataLoader(testset, batch_size=128, shuffle=False)
model = Net().to(device)
nllloss = nn.NLLLoss().to(device)
loss_weight = 1
centerloss = CenterLoss(10, 2).to(device)
optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9, weight_decay=0.0005)
sheduler = lr_scheduler.StepLR(optimizer,20,gamma=0.8)

# optimzer4center
optimzer4center = optim.SGD(centerloss.parameters(), lr =0.5)
total = 0.0
correct = 0.0
max_corr = 0.0
for epoch in range(100):
    sheduler.step()
    train(num,epoch+1)
