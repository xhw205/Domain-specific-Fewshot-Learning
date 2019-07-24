import torch.nn as nn
import torch
import miniPPlankton.data.fewdataloader as tg
from tqdm import tqdm
from miniPPlankton.net.resnet18 import EmbeddingNetwork
from miniPPlankton.net.attNet import CNNEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingNetwork().to(device)  #backbone
model.load_state_dict(torch.load('./models/c.pkl'))
CNNEncoder = CNNEncoder().to(device) # 84*84 focus-areas
CNNEncoder.load_state_dict(torch.load('./models/att.pkl'))
CNNEncoder.eval()
model.eval()
maxacc = 0.0
metatrain_folders, metatest_folders = tg.mini_imagenet_folders()
correct = 0.0
total = 0.0
CLASS_NUM = 10
SUPPORT_NUM_PER_CLASS = 1
QUERY_NUM_PER_CLASS = 15
with torch.no_grad():
    for i in tqdm(range(100)):
        task = tg.MiniImagenetTask(metatest_folders, CLASS_NUM, SUPPORT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
        support_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SUPPORT_NUM_PER_CLASS, split="train",
                                                              shuffle=False)
        query_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=5, split="test",
                                                            shuffle=False)
        support, support_labels, s_att = support_dataloader.__iter__().next()
        support_features,_,_ = model(support.to(device))
        support_features = support_features.view(CLASS_NUM, SUPPORT_NUM_PER_CLASS, -1)
        support_features = torch.sum(support_features, 1) / SUPPORT_NUM_PER_CLASS
        support_features = support_features

        s_att_features = CNNEncoder(s_att.to(device))
        s_att_features = s_att_features.view(CLASS_NUM, SUPPORT_NUM_PER_CLASS, -1)
        s_att_features = torch.sum(s_att_features, 1) / SUPPORT_NUM_PER_CLASS
        # support_features = torch.cat([support_features, s_att_features], 1) #with_att
        eps = 1e-10
        sum_support = torch.sum(torch.pow(support_features, 2), 1)
        s_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
        for query, query_labels,q_att in query_dataloader:
            query_labels = query_labels.to(device)
            batch = query_labels.size(0)
            query_features,_,_ = model(query.to(device))
            q_att_features = CNNEncoder(q_att.to(device))
            # query_features = torch.cat([query_features,q_att_features],1)
            dot_product = query_features.matmul(support_features.t())
            cosine_similarity = dot_product * s_magnitude
            softmax = nn.Softmax()
            softmax_similarities = softmax(cosine_similarity)
            _, preds = torch.max(softmax_similarities, 1)
            correct += torch.sum(preds == query_labels.data)
            total += batch
accuracy = correct.double() / total
print ("acc:",accuracy)
