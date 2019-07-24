#coding=utf-8
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms,models
from experiments.cnloss import CenterLoss
import data.cubloader as task
from net.extractor import EmbeddingNetwork
from net.Orderextractor import OEmbeddingNetwork
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ]),
    'att': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
datapath = '/home/ws/datasets/CUB_200_2011/'  #your own path
train_dataset = task.ImageLoader(
    datapath,
    transform=data_transforms['train'],
    att_transform=data_transforms['att'],
    train=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True,
    num_workers=4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_acc = 0.0
def supp_idxs(t, c):
    return t.eq(c).nonzero().squeeze(1)
def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    return torch.pow(x - y, 2).sum(2)

def train(model, criterion, optimizer, scheduler,num_epochs=50,CNloss=True):
    num = 0
    best_loss = 1000000000.0
    for epoch in range(num_epochs):
        scheduler.step()
        exp_center.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        total_batch = 0
        for idx, batch in enumerate(train_loader):
            inputs, labels, _ = batch
            total_batch+=inputs.size(0)
            num += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            optimzer4center.zero_grad()
            _, ip1, outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            c, centers = centerloss(labels, ip1)
            loss = criterion(outputs, labels)
            if CNloss:
                loss = loss + 1.0*c
                cls = torch.unique(labels.cpu())
                s_idx = [supp_idxs(labels.cpu(), i) for i in cls]
                pre = torch.stack([ip1[idx_list].mean(0) for idx_list in s_idx])
                dists = euclidean_dist(pre, centers)
                log_p_y = F.log_softmax(-dists, dim=1).view(len(s_idx), 100)
                _, y_hat = torch.max(log_p_y.data,1)
                cls = cls.view(cls.size(0),-1).cuda()
                loss_val = -log_p_y.gather(1, cls.long()).squeeze().view(-1).mean()
                loss = loss + 0.6*loss_val
            loss.backward()
            optimizer.step()
            optimzer4center.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / total_batch
        print('{} Loss: {:.4f}'.format('train', epoch_loss))
        if epoch_loss < best_loss:
            print("save!")
            best_loss = epoch_loss
            torch.save(model.state_dict(), "./models/temp.pkl")

if __name__ == "__main__":
    criterion = nn.CrossEntropyLoss()
    centerloss = CenterLoss(100, 2).to(device)
    optimzer4center = torch.optim.SGD(centerloss.parameters(), lr=0.5)
    exp_center = torch.optim.lr_scheduler.StepLR(optimzer4center, step_size=20, gamma=0.5)
    CNloss = True
    embedding = OEmbeddingNetwork().to(device)
    optimizer = torch.optim.SGD(embedding.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    model= train(embedding, criterion, optimizer, exp_lr_scheduler, num_epochs=100, CNloss=CNloss)
