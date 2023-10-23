import torch
import random
import math

import torch.nn as nn
from torchvision import transforms
from torch.optim.lr_scheduler import _LRScheduler
from PIL import ImageOps, ImageFilter

from clip import clip
# from eda import *

def load_clip_to_cpu(backbone_name):

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class CustomCLIP(nn.Module):
    def __init__(self, clip_model, class_nums):
        super().__init__()
        self.model = clip_model
        self.outputdim = clip_model.visual.output_dim

        self.image_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.image_classifier.weight_g.data.fill_(1)
        self.image_classifier.weight_g.requires_grad = False

        self.text_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.text_classifier.weight_g.data.fill_(1)
        self.text_classifier.weight_g.requires_grad = False

    def encode_image(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, tokens):
        text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images, text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)

        logits_image = self.image_classifier(image_features)
        logits_text = self.text_classifier(text_features)

        return logits_image, logits_text, image_features, text_features


class Solarize(object):
    def __call__(self, img):
        return ImageOps.solarize(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_dino_aug(size=224, scale=(0.2, 1.0), gaussian=0.5, solarize=0.5):
    augs = [
        transforms.RandomResizedCrop(size, scale=scale),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]

    if gaussian > 0:
        augs.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=gaussian))
    
    if solarize > 0:
        augs.append(transforms.RandomApply([Solarize()], p=solarize))
    
    augs.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(augs)

class ImageViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
        self.simgcd_transform = transforms.Compose([
                transforms.Resize(int(224 / 0.875), 3),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor((0.485, 0.456, 0.406)),
                    std=torch.tensor((0.229, 0.224, 0.225)))
            ])
        self.strong_transform = get_dino_aug(size=224, scale=(0.140, 1.000), gaussian=0.5, solarize=0.1)
        # interpolation = 3
        # crop_pct = 0.875
        # image_size = 224
        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        # self.strong_transform = transforms.Compose([transforms.Resize(int(image_size / crop_pct), interpolation),
        #                                             transforms.RandomCrop(image_size),
        #                                             transforms.RandomHorizontalFlip(p=0.5),
        #                                             transforms.ColorJitter(),
        #                                             transforms.ToTensor(),
        #                                             transforms.Normalize(
        #                                                 mean=torch.tensor(mean),
        #                                                 std=torch.tensor(std))
        #                                         ])

    def __call__(self, x):
        # if not isinstance(self.base_transform, list):
        #     return [self.base_transform(x), self.base_transform(x), self.strong_transform(x)]
        # else:
        #     return [self.base_transform(x), self.base_transform(x), self.strong_transform(x)]
        # if not isinstance(self.base_transform, list):
        #     return [self.base_transform(x), self.base_transform(x), self.simgcd_transform(x)]
        # else:
        #     return [self.base_transform(x), self.base_transform(x), self.simgcd_transform(x)]
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]

class TextViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, alpha_sr=0, alpha_ri=0, alpha_rs=0, alpha_rd=0, seed=0):
        # self.augmenter_spelling = naw.SpellingAug(aug_p=0.05)
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.alpha_rd = alpha_rd
        self.seed = seed

    def __call__(self, text):
        # return [text, self.augmenter_spelling.augment(text)]
        # aug_text = eda(text, alpha_sr=self.alpha_sr, alpha_ri=self.alpha_ri, alpha_rs=self.alpha_rs, p_rd=self.alpha_rd, num_aug=2, seed=self.seed)
        return [text, text]

# lr scheduler for classifier
class CustomCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, classifier_params, T_max, eta_min=0, last_epoch=-1):
        self.classifier_params_ids = set(map(id, classifier_params))
        self.T_max = T_max
        self.eta_min = eta_min
        self.classifier_lr = optimizer.param_groups[0]['lr']
        self.base_lr = optimizer.param_groups[1]['lr']
        super(CustomCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (self.classifier_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            if any(id(p) in self.classifier_params_ids for p in param_group['params']) 
            else self.base_lr
            for param_group in self.optimizer.param_groups
        ]