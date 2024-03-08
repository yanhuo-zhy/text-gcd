from .oxford_iiit_pet import OxfordIIITPet
from copy import deepcopy
import numpy as np
from typing import Any, Tuple
from PIL import Image

from data.data_utils import subsample_instances
from utils import process_file, construct_text
from clip import clip

from config import oxford_pet_root, oxford_pet_tag_root

class OxfordPet_Base(OxfordIIITPet):

    def __init__(self, root=oxford_pet_root, split='trainval', transform=None, target_transform=None, download=False):

        super(OxfordPet_Base, self).__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.data = np.array(self._images)
        self.targets = np.array(self._labels)
        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = Image.open(self.data[idx]).convert("RGB")

        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self.targets[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

class OxfordPet_LENS(OxfordPet_Base):
    def __init__(self, root=oxford_pet_root, split='trainval', transform=None, target_transform=None, download=False):
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)
        
    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return {"image": img, "target": label, "image_id": str(uq_idx), "id": uq_idx}

class OxfordPetDataset(OxfordPet_Base):
    def __init__(self, tag_root, root=oxford_pet_root, split='trainval', transform=None, target_transform=None, text_transform=None, download=False):
        self.tag = process_file(tag_root)
        self.text_transform = text_transform        
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform, download=download)

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

    def __getitem__(self, item):
        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        tag_text = construct_text('A photo of a pet, ', self.tag[str(uq_idx)])

        if self.text_transform is not None:
            tag_text = self.text_transform(tag_text)
            tag_token = [self.safe_tokenize(t) for t in tag_text]
        else:
            tag_token = self.safe_tokenize(tag_text)
        return img, label, str(uq_idx), tag_token, uq_idx
    
def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:
        dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:

        return None


def subsample_classes(dataset, include_classes=(0, 1, 8, 9)):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_oxford_pets_datasets(train_transform, 
                          test_transform,
                          text_transform, 
                          tag_root=oxford_pet_tag_root, 
                          train_classes=(0, 1, 8, 9),
                          prop_train_labels=0.8, 
                          split_train_val=False, 
                          seed=0):

    np.random.seed(seed)

    # Init entire training set
    whole_training_set = OxfordPetDataset(tag_root=tag_root, root=oxford_pet_root, transform=train_transform, text_transform=text_transform)

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
    test_dataset = OxfordPetDataset(tag_root=tag_root, root=oxford_pet_root, transform=test_transform, text_transform=None, split='test')

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

    x = get_oxford_pets_datasets(None, None, split_train_val=False,
                         train_classes=range(10), prop_train_labels=0.5)

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