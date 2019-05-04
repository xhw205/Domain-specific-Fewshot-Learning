import torch
from PIL import Image
import os
import pandas as pd
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, att_transform=None,  base_one=False, train=False, num_classes=100, num_train_sample=0,
                 novel_only=False, test_only=False, aug=False,
                 loader=pil_loader):
        img_folder = os.path.join(root, "images")
        img_paths = pd.read_csv(os.path.join(root, "images.txt"), sep=" ", header=None, names=['idx', 'path'])
        img_labels = pd.read_csv(os.path.join(root, "image_class_labels.txt"), sep=" ", header=None,
                                 names=['idx', 'label'])
        train_test_split = pd.read_csv(os.path.join(root, "train_test_split.txt"), sep=" ", header=None,
                                       names=['idx', 'train_flag'])
        data = pd.concat([img_paths, img_labels, train_test_split], axis=1)
        data = data[data['train_flag'] == train]
        data['label'] = data['label'] - 1

        # split dataset
        data = data[data['label'] < num_classes]
        base_data = data[data['label'] < 100]
        novel_data = data[data['label'] >= 100]

        # sampling from novel classes
        if num_train_sample != 0:
            b = 5
            novel_data = novel_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[5:5+num_train_sample])
            base_one_data = base_data.groupby('label', group_keys=False).apply(lambda x: x.iloc[5:5+num_train_sample])
        # whether only return data of novel classes
        if novel_only: # Train novel classes
            data = novel_data
        else:
            data = pd.concat([base_data, novel_data]) #Test across all class

        if not novel_only and test_only:# Test across base class
            data = base_data

        if novel_only is True and base_one is True: #Train on only one sample
            data = pd.concat([base_one_data, novel_data])
        if aug:
            tmp_data = pd.DataFrame()
            for i in range(5):
                tmp_data = pd.concat([tmp_data, data])
            data = tmp_data
        imgs = data.reset_index(drop=True)

        if len(imgs) == 0:
            raise (RuntimeError("no csv file"))
        self.root = img_folder
        self.imgs = imgs
        self.transform = transform
        self.att_transform = att_transform

        self.loader = loader
        self.train = train

    def __getitem__(self, index):

        item = self.imgs.iloc[index]
        file_path = item['path']
        target = item['label']
        path = os.path.join(self.root, file_path)
        items = path.split('/')
        items[4] = 'CUB_Attention'
        attpath = os.path.join('/',*items)
        img = self.loader(path)
        # print(path)
        att = Image.open(attpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.att_transform is not  None:
            att = self.att_transform(att)

        return img, target, att
    def __len__(self):
        return len(self.imgs)
