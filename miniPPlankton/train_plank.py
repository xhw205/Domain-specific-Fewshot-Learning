import torchvision.datasets as datasets
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from net.resnet18 import EmbeddingNetwork
from experiments.ccloss import CenterLoss
root = '/home/ws/datasets/phytoplankton/fewshot/SmallTrain/' #your own path for miniPPlankton
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]),
}
def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

model = EmbeddingNetwork().cuda()
image_datasets_train = datasets.ImageFolder(os.path.join(root), data_transforms['train'])
dataloaders_train = torch.utils.data.DataLoader(image_datasets_train, batch_size=32,
                                             shuffle=True, num_workers=8)
dataset_sizes_train = len(image_datasets_train)
criterion = nn.CrossEntropyLoss()
optimizer =  torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4) 
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
centerloss = CenterLoss(10,2).cuda()
optimzer4center = torch.optim.SGD(centerloss.parameters(), lr=0.5)
exp_center = torch.optim.lr_scheduler.StepLR(optimzer4center, step_size=20, gamma=0.5) #20better
maxacc = 0.0
best_loss = 10000000.0
def supp_idxs(t, c):
    return t.eq(c).nonzero().squeeze(1)
num = 0
for epoch in range(50):
    exp_lr_scheduler.step()
    exp_center.step()
    running_loss = 0.0
    totalloss = 0
    ip1_loader = []
    idx_loader = []
    for inputs, targets in dataloaders_train:
        inputs = inputs.cuda()
        targets = targets.cuda()
        optimizer.zero_grad()
        optimzer4center.zero_grad()
        avg, ip1, outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        c, centers = centerloss(targets, ip1)
        loss = criterion(outputs, targets) + 1.0 * c
        cls = torch.unique(targets.cpu())
        s_idx = [supp_idxs(targets.cpu(), i) for i in cls] # idx
        pre = torch.stack([ip1[idx_list].mean(0) for idx_list in s_idx])
        dists = euclidean_dist(pre, centers)
        log_p_y = F.log_softmax(-dists, dim=1).view(len(s_idx), 10)
        _, y_hat = torch.max(log_p_y.data,1)
        cls = cls.view(cls.size(0),-1).cuda()
        loss_val = -log_p_y.gather(1, cls.long()).squeeze().view(-1).mean()
        loss = loss+0.6*loss_val  #0.6
        num+=1
        loss.backward()
        optimizer.step()
        optimzer4center.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / dataset_sizes_train
    print("loss:{}".format(epoch_loss))
    if epoch_loss < best_loss:
        print("save!")
        best_loss = epoch_loss
        torch.save(model.state_dict(), "./models/plank.pkl")

