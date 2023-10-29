import os
import requests
from lens import Lens, LensProcessor
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse

from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from data.cub import CustomCub2011_LENS
from data.stanford_cars import CarsDataset_LENS
from data.fgvc_aircraft import FGVCAircraft_LENS
from data.cifar import CustomCIFAR10_LENS, CustomCIFAR100_LENS
from data.herbarium_19 import HerbariumDataset19_LENS
from data.imagenet import ImageNetDataset_LENS
from data.oxford_pets import OxfordPet_LENS
from data.oxford_flowers import OxfordFlowers_LENS
from data.food101 import Food101_LENS

dataset_map = {
    'cub': CustomCub2011_LENS,
    'scars': CarsDataset_LENS,
    'aircraft': FGVCAircraft_LENS,
    'cifar10': CustomCIFAR10_LENS,
    'cifar100': CustomCIFAR100_LENS,
    'herbarium_19': HerbariumDataset19_LENS,
    'imagenet': ImageNetDataset_LENS,
    'pets' : OxfordPet_LENS,
    'flowers' : OxfordFlowers_LENS,
    'food' : Food101_LENS
}

class LensDataset:
    def __init__(
        self,
        ds: Dataset,
        questions: Optional[List[str]] = None,
        processor: Optional[LensProcessor] = None,
    ):
        self.ds = ds
        self.processor = processor
        self.questions = questions

    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        id = self.ds[idx]["id"]
        try:
            question = self.ds[idx]["question"]
        except:
            pass
        try:
            question = self.questions[idx]
        except:
            question = ""
        outputs = self.processor([image], question)
        return self.ds[idx]["image_id"],{
            "id": torch.tensor(id, dtype=torch.int32),
            "clip_image": outputs["clip_image"].squeeze(0),
            "blip_image": outputs["blip_image"].squeeze(0),
            "blip_input_ids": outputs["blip_input_ids"].squeeze(0),
            "questions": outputs["questions"],
        }

    def __len__(self):
        return len(self.ds)

def generate_tags_attributes_for_one_batch(dataloader, lens):
    tags_attributes_list = []

    # 获取一个batch的数据
    batch = next(iter(dataloader))
    image_ids, samples = batch
    
    with torch.no_grad():
        # questions = ["What is the name of this kind of bird?"] * len(imgs)
        # samples = processor(imgs, questions)
        outputs = lens(samples)
        
        for i, output in enumerate(outputs["prompts"]):
            prompts = output
                
            # 解析tags和attributes
            lines = prompts.split('\n')
            tags_start = lines.index('Tags:') + 1
            attributes_start = lines.index('Attributes:') + 1
            captions_start = lines.index('Captions:')
            
            tags = ["<tag>" + line.replace('-', '').strip() for line in lines[tags_start:attributes_start-1]][:5]
            attributes = ["<attribute>" + line.replace('-', '').strip() for line in lines[attributes_start:captions_start]][:5]
    
            # 整合结果
            result_str = image_ids[i] + "|||" + "|||".join(tags) + "|||" + "|||".join(attributes)
            tags_attributes_list.append(result_str)

    return tags_attributes_list

def generate_tags_attributes_for_batches(dataloader, lens, output_file, start_idx=0):
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches", initial=start_idx, total=len(dataloader))):
        if batch_idx < start_idx:
            continue  # 跳过已经处理过的batches

        image_ids, samples = batch
        
        with torch.no_grad():
            outputs = lens(samples)
            tags_attributes_list = []

            for i, output in enumerate(outputs["prompts"]):
                prompts = output

                # 解析tags和attributes
                lines = prompts.split('\n')
                tags_start = lines.index('Tags:') + 1
                attributes_start = lines.index('Attributes:') + 1
                captions_start = lines.index('Captions:')

                tags = ["<tag>" + line.replace('-', '').strip() for line in lines[tags_start:attributes_start-1]][:5]
                attributes = ["<attribute>" + line.replace('-', '').strip() for line in lines[attributes_start:captions_start]][:5]

                # 整合结果
                result_str = image_ids[i] + "|||" + "|||".join(tags) + "|||" + "|||".join(attributes)
                tags_attributes_list.append(result_str)

            # 写入文件
            save_to_file(tags_attributes_list, output_file)

            # 更新当前处理到的batch索引
            with open(output_file + '.idx', 'w') as f_idx:
                f_idx.write(str(batch_idx + 1))


def save_to_file(data, filepath, mode='a'):
    with open(filepath, mode) as f:
        for line in data:
            f.write(line + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=400, type=int)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--dataset_name', type=str, default='imagenet', help='options: cifar10, cifar100, imagenet, cub, scars, aircraft, herbarium_19, pets, flowers, food')
    parser.add_argument('--data_root', type=str, default='/wang_hp/zhy/data/stanford_cars')
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=20)   

    args = parser.parse_args()
    dataset = dataset_map[args.dataset_name]()
    lens = Lens()
    processor = LensProcessor()
    lendataset = LensDataset(ds=dataset, processor=processor)
    print(len(lendataset))  

    dataloader = DataLoader(lendataset, batch_size=args.batch_size, shuffle=False)

    output_file = f'tag/{args.dataset_name}_tags_attributes.txt'
    start_idx = 0
    if os.path.exists(output_file + '.idx'):
        with open(output_file + '.idx', 'r') as f_idx:
            start_idx = int(f_idx.read().strip())

    dataloader = DataLoader(lendataset, batch_size=args.batch_size, shuffle=False)

    generate_tags_attributes_for_batches(dataloader, lens, output_file, start_idx=start_idx)