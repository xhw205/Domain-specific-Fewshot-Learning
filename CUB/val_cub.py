import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import data.cubloader
from net.Orderextractor import OEmbeddingNetwork
from experiments.cosineclassifer import CosineClassifier
from utils.parser import get_arg
args = get_arg()

SUPPORT_NUM = args.val_support_num
novelonly = True
withAtt = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    model_path = "./models/temp.pkl"
    datapath = args.datapath
    if withAtt:
        print("Foucus area could help you!")
    print('==> Reading from model checkpoint..')
    embedding = OEmbeddingNetwork().to(device)
    embedding.load_state_dict(torch.load(model_path))
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
    novel_dataset = data.cubloader.ImageLoader(
        0,
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
        data.cubloader.ImageLoader(0,datapath, novel_trasforms
        , att_trasforms, num_classes=200, novel_only=novelonly),
        batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True)
    cosineclassifer = CosineClassifier(with_att=withAtt, novel_only=novelonly)
    acc = fewcub(cosineclassifer, novel_loader, val_loader, embedding)
    return acc

def fewcub(classifier, novel_loader, val_loader, model):
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
            query_features,ip1,_ = model(input)
            att_f,_,_ = model(att)
            similarities = classifier(basefeat=None, basefeat_att=None,
                       supportfeat=output_stack, supportfeat_att=output_cat, queryfeat=query_features, queryfeat_att=att_f)
            _, preds = torch.max(similarities, 1)
            if novelonly:
                preds = preds + torch.tensor(100).cuda()
            correct += torch.sum((preds) == target.data)
        print("Acc:%.4f"%(correct.double()/total))
        return correct.double()/total
if __name__ == '__main__':
    main()