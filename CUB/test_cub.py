import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import data.cubloader
from net.Orderextractor import OEmbeddingNetwork
from CUB.experiments.cosineclassifer import CosineClassifier

######## Args #########
SUPPORT_NUM = 1
novelonly = True
withAtt = False
model_path = "./models/closs_high.pkl"
baseclass_weight = './models/baseO.pkl'
baseclass_att_weight = './models/baseAttOrder.pkl'
data_path = '/home/ws/datasets/CUB_200_2011/'
######## Args #########



if withAtt:
    print("Foucus area could help you!")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('==> Reading from model checkpoint..')
embedding = OEmbeddingNetwork().to(device)
embedding.load_state_dict(torch.load(model_path))
def main():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    novel_trasforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
    ])
    att_trasforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    datapath = data_path
    novel_dataset = data.cubloader.ImageLoader(
        datapath,
        novel_trasforms,
        att_transform=att_trasforms,
        train=True, num_classes=200,
        num_train_sample=SUPPORT_NUM,
        novel_only=True, aug=False)
    novel_loader = torch.utils.data.DataLoader(
        novel_dataset, batch_size=100, shuffle=False,
        num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        data.cubloader.ImageLoader(datapath, novel_trasforms
        , att_trasforms, num_classes=200, novel_only=novelonly),
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)
    cosineclassifer = CosineClassifier(with_att=withAtt, novel_only=novelonly)
    acc = fewcub(cosineclassifer, novel_loader, val_loader, embedding)

def fewcub(classifier, novel_loader, val_loader, model):
    basetrain, basetrainatt = None, None
    if novelonly is False:
        basetrain = torch.load(baseclass_weight) #base train weight
        basetrainatt = torch.load(baseclass_att_weight) #base train of focus-area's weight
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target, att) in enumerate(novel_loader):
            input, target, att = input.to(device), target.to(device), att.to(device)
            output,_,_ = model(input)
            o_att,_,_ = model(att)
            if batch_idx == 0:
                output_stack = output
                output_cat = o_att
            else:
                output_stack = torch.cat((output_stack, output),0)
                output_cat = torch.cat((output_cat,o_att),0)
        output_stack = torch.sum(output_stack.view(100,SUPPORT_NUM,-1),1)/SUPPORT_NUM
        output_cat = torch.sum(output_cat.view(100,SUPPORT_NUM, -1), 1) / SUPPORT_NUM
        correct = 0.0
        total = 0.0
        for batch_idx, (input, target, att) in enumerate(val_loader):
            total+=target.size(0)
            input, att, target = input.to(device), att.to(device), target.to(device)
            query_features,_,_ = model(input)
            att_f,_,_ = model(att)
            similarities = classifier(basefeat=basetrain, basefeat_att=basetrainatt,
                       supportfeat=output_stack, supportfeat_att=output_cat, queryfeat=query_features, queryfeat_att=att_f)
            _, preds = torch.max(similarities, 1)
            if novelonly:
                preds = preds + torch.tensor(100).cuda()
            correct += torch.sum((preds) == target.data)
        print("Acc:%.4f"%(correct.double()/total))
        return correct.double()/total
if __name__ == '__main__':
    main()