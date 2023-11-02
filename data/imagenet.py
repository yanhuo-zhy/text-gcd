import torchvision
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
from utils import process_file, construct_text
from config import imagenet_root, imagenet_tag_root

from clip import clip

class ImageNetDataset_Base(torchvision.datasets.ImageFolder):

    def __init__(self, root=os.path.join(imagenet_root, 'train'), transform=None):

        super(ImageNetDataset_Base, self).__init__(root=root, transform=transform)

        self.uq_idxs = np.array(range(len(self)))

class ImageNetDataset_LENS(ImageNetDataset_Base):

    def __init__(self, root=os.path.join(imagenet_root, 'train'), transform=None):
        super().__init__(root=root, transform=transform)

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]
        path, _ = self.samples[item]
        image_id = os.path.basename(path)

        return {"image": img, "target": label, "image_id": image_id, "id": uq_idx}

class ImageNetDataset(ImageNetDataset_Base):
    def __init__(self, tag_root, root=os.path.join(imagenet_root, 'train'), transform=None, text_transform=None):
        self.tag = process_file(tag_root)
        self.text_transform = text_transform  
        super().__init__(root=root, transform=transform)

    def safe_tokenize(self, text):
        while True:
            try:
                token = clip.tokenize(text)
                return token
            except RuntimeError:
                words = text.split()
                if len(words) <= 1:  
                    return None
                text = " ".join(words[:-1])  

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        uq_idx = self.uq_idxs[idx]
        path, _ = self.samples[idx]
        image_id = os.path.basename(path)
        try:
            tag_text = construct_text('', self.tag[image_id])
        except:
            print("without id")
            tag_text = "A photo of an object"


        if self.text_transform is not None:
            tag_text = self.text_transform(tag_text)
            tag_token = [self.safe_tokenize(t) for t in tag_text]
        else:
            tag_token = self.safe_tokenize(tag_text)
        
        return img, label, image_id, tag_token, uq_idx

def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_imagenet_100_datasets(train_transform, 
                              test_transform, 
                              text_transform,
                              tag_root=imagenet_tag_root, 
                              train_classes=range(80),
                              prop_train_labels=0.8, 
                              split_train_val=False, 
                              seed=0):

    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetDataset(tag_root=tag_root, root=os.path.join(imagenet_root, 'train'), transform=train_transform, text_transform=text_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetDataset(tag_root=tag_root, root=os.path.join(imagenet_root, 'val'), transform=test_transform, text_transform=text_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets


def get_imagenet_1k_datasets(train_transform, 
                             test_transform, 
                             text_transform,
                             tag_root=imagenet_tag_root,
                             train_classes=range(500),
                             prop_train_labels=0.5, 
                             split_train_val=False, 
                             seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = ImageNetDataset(tag_root=tag_root, root=os.path.join(imagenet_root, 'train'), transform=train_transform, text_transform=text_transform)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = ImageNetDataset(tag_root=tag_root, root=os.path.join(imagenet_root, 'val'), transform=test_transform, text_transform=None)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets



if __name__ == '__main__':

    x = get_imagenet_100_datasets(None, None, split_train_val=False,
                               train_classes=range(50), prop_train_labels=0.5)

    print('Printing lens...')
    for k, v in x.items():
        if v is not None:
            print(f'{k}: {len(v)}')

    print('Printing labelled and unlabelled overlap...')
    print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
    print('Printing total instances in train...')
    print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

    print(f'Num Labelled Classes: {len(set(x["train_labelled"].targets))}')
    print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].targets))}')
    print(f'Len labelled set: {len(x["train_labelled"])}')
    print(f'Len unlabelled set: {len(x["train_unlabelled"])}')