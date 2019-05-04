import torch
import torch.nn.functional as F
import numpy as np
from utils.util import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
import torchvision.utils as vutils
np.set_printoptions(threshold=np.inf)
class GradCAM(object):

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()
        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        target_layer = self.model_arch.layer4
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)
        self.model_arch(torch.zeros(1, 3, *(model_dict['input_size']), device="cuda"))


    def forward(self, input, class_idx=None, retain_graph=False):

        b, c, h, w = input.size()

        _,_,logit = self.model_arch(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        mask = saliency_map.ge(0.6).float()
        mask_temp = mask.cpu().squeeze(0).squeeze(0).numpy()
        itemindex = np.argwhere(mask_temp == 1.0)
        y_min, y_max = itemindex[:, :1].min(), itemindex[:, :1].max()
        x_min, x_max = itemindex[:, 1:].min(), itemindex[:, 1:].max()
        box = (x_min, y_min, x_max, y_max)
        return box, saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
#