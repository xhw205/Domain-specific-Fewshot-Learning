import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as tvt
from CUB.utils.gradcam import GradCAM
from CUB.utils.util import visualize_cam, Normalize
from torchvision.utils import make_grid, save_image
from CUB.net.Orderextractor import  OEmbeddingNetwork
ToPil = lambda x:Image.open(x).convert('RGB')
transforms = tvt.Compose([
        tvt.Resize((224,224)),
        tvt.ToTensor(),
        tvt.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
resnet = OEmbeddingNetwork().cuda()
resnet.load_state_dict(torch.load( './models/closs_high.pkl'))
resnet.eval()
cam_dict = dict()
resnet_model_dict = dict(type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
gradcam = GradCAM(resnet_model_dict, True)
imgpaths = [] #All_images_path

root = '/home/ws/datasets/CUB_200_2011/images/'
newroot = '/home/ws/datasets/CUB_temp'
os.makedirs(newroot, exist_ok=True)

for root,file,items in os.walk(root):
    if items != []:
        for i in items:
            imgpaths.append(os.path.join(root,i))

normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
for img_path in imgpaths:
    filename = img_path.split('/')[-2]
    imgname = img_path.split('/')[-1]
    newroot_path = os.path.join(newroot, filename) #the path to save focus-area
    if not os.path.isdir(newroot_path):
        os.mkdir(newroot_path)
    pil_img = Image.open(img_path).convert('RGB')
    torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.interpolate(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
    normed_torch_img = normalizer(torch_img)
    box, mask, _ = gradcam(normed_torch_img)
    print(os.path.join(newroot_path, imgname))
    tvt.Resize((224,224))(pil_img).crop(box).save(os.path.join(newroot_path, imgname))