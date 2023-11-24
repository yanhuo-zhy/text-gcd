import os
import requests
import torch
from torchvision import transforms
from tqdm import tqdm
import argparse
from datasets import Dataset, load_dataset

from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import sys
sys.path.append('..')
from data.cub import CustomCub2011_LENS
from data.stanford_cars import CarsDataset_LENS
from data.fgvc_aircraft import FGVCAircraft_LENS
from data.cifar import CustomCIFAR10_LENS, CustomCIFAR100_LENS
from data.herbarium_19 import HerbariumDataset19_LENS
from data.imagenet import ImageNetDataset_LENS
# from data.oxford_pets import OxfordPet_LENS
# from data.oxford_flowers import OxfordFlowers_LENS
# from data.food101 import Food101_LENS

from transformers import Blip2Processor, Blip2ForConditionalGeneration

dataset_map = {
    'cub': CustomCub2011_LENS,
    'scars': CarsDataset_LENS,
    'aircraft': FGVCAircraft_LENS,
    'cifar10': CustomCIFAR10_LENS,
    'cifar100': CustomCIFAR100_LENS,
    'herbarium_19': HerbariumDataset19_LENS,
    'imagenet': ImageNetDataset_LENS,
    # 'pets' : OxfordPet_LENS,
    # 'flowers' : OxfordFlowers_LENS,
    # 'food' : Food101_LENS
}

def flatten(l):
    return [item for sublist in l for item in sublist]


class Lens(nn.Module):
    def __init__(
        self,
        blip_name: str = "/wang_hp/zhy/LEN/Salesforce/blip-image-captioning-large",
        load_8bit: bool = False,
        device: torch.device = 'cuda',
    ):
        super().__init__()
        # Load Base models
        self.device = device

        self.blip_name = blip_name

        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

    def __call__(
        self,
        samples: dict,
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 5,
        return_intensive_captions: bool = True,
    ):

        samples = self.forward_tag(
            samples,
            max_length=max_length,
            min_length=min_length,
            top_k=top_k,
            num_captions=num_captions,
        )
        samples = self.forward_attribute(
            samples,
            max_length=max_length,
            min_length=min_length,
            top_k=top_k,
            num_captions=num_captions,
        )
        return samples


    def forward_tag(
        self,
        samples: dict,
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
    ):
        pixel_values = samples["blip_image_name"].to(self.device, self.blip_model.dtype)
        input_ids = samples["blip_input_ids_name"].to(self.device)
        caption_ids = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=1,
            top_k=top_k,
            repetition_penalty=1,
            num_return_sequences=num_captions,
        )

        captions_text = self.blip_processor.batch_decode(
            caption_ids, skip_special_tokens=True
        )
        captions_text = [caption.strip() for caption in captions_text]
        captions_text = [
            captions_text[i : i + num_captions]
            for i in range(0, len(captions_text), num_captions)
        ]
        samples["tags"] = captions_text
        return samples
    
    def forward_attribute(
        self,
        samples: dict,
        max_length: int = 30,
        min_length: int = 10,
        top_k: int = 50,
        num_captions: int = 10,
    ):
        pixel_values = samples["blip_image_feature"].to(self.device, self.blip_model.dtype)
        input_ids = samples["blip_input_ids_feature"].to(self.device)
        caption_ids = self.blip_model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_p=1,
            top_k=top_k,
            repetition_penalty=1,
            num_return_sequences=num_captions,
        )

        captions_text = self.blip_processor.batch_decode(
            caption_ids, skip_special_tokens=True
        )
        captions_text = [caption.strip() for caption in captions_text]
        captions_text = [
            captions_text[i : i + num_captions]
            for i in range(0, len(captions_text), num_captions)
        ]
        samples["attributes"] = captions_text
        return samples

class LensProcessor:
    def __init__(
        self,
        blip_name: str = "/wang_hp/zhy/LEN/Salesforce/blip-image-captioning-large",
    ):
        self.blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def __call__(self, images):

        outputs_name = self.blip_processor(
            images=images, text=["The name of this bird is"] * len(images), return_tensors="pt"
        )
        blip_image_name = outputs_name["pixel_values"]
        blip_input_ids_name = outputs_name["input_ids"]

        outputs_feature = self.blip_processor(
            images=images, text=["The feature of this bird is"] * len(images), return_tensors="pt"
        )
        blip_image_feature = outputs_feature["pixel_values"]
        blip_input_ids_feature = outputs_feature["input_ids"]
        return {
            "blip_image_name": blip_image_name,
            "blip_input_ids_name": blip_input_ids_name,
            "blip_image_feature": blip_image_feature,
            "blip_input_ids_feature": blip_input_ids_feature,
        }


class LensDataset:
    def __init__(
        self,
        ds: Dataset,
        processor,
    ):
        self.ds = ds
        self.processor = processor


    def __getitem__(self, idx):
        image = self.ds[idx]["image"]
        id = self.ds[idx]["id"]

        outputs = self.processor([image])
        return self.ds[idx]["image_id"], {
            "id": torch.tensor(id, dtype=torch.int32),
            "blip_image_name": outputs["blip_image_name"].squeeze(0),
            "blip_input_ids_name": outputs["blip_input_ids_name"].squeeze(0),
            "blip_image_feature": outputs["blip_image_feature"].squeeze(0),
            "blip_input_ids_feature": outputs["blip_input_ids_feature"].squeeze(0),
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
        
        for i, output in enumerate(outputs['tags']):
            tags_i = ["<tag>" + tag for tag in output][:5]
            attributes_i = ["<attribute>" + attribute for attribute in outputs['attributes'][i]][:5]
            result_str = image_ids[i] + "|" + "|".join(tags_i) + "|" + "|".join(attributes_i)
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

            for i, output in enumerate(outputs['tags']):
                tags_i = ["<tag>" + tag for tag in output][:5]
                attributes_i = ["<attribute>" + attribute for attribute in outputs['attributes'][i]][:5]
                result_str = image_ids[i] + "|" + "|".join(tags_i) + "|" + "|".join(attributes_i)
                tags_attributes_list.append(result_str)

            # 写入文件
            save_to_file(tags_attributes_list, output_file)

            # 更新当前处理到的batch索引
            with open(output_file + '.idx', 'w') as f_idx:
                f_idx.write(str(batch_idx + 1))
        break

def save_to_file(data, filepath, mode='a'):
    with open(filepath, mode) as f:
        for line in data:
            f.write(line + '\n')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--dataset_name', type=str, default='cub', help='options: cifar10, cifar100, imagenet, cub, scars, aircraft, herbarium_19, pets, flowers, food')
    parser.add_argument('--data_root', type=str, default='/wang_hp/zhy/data/stanford_cars')
    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=20)   

    args = parser.parse_args()
    dataset = dataset_map[args.dataset_name]()

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # model = BlipForConditionalGeneration.from_pretrained("/wang_hp/zhy/LEN/Salesforce/blip-image-captioning-large").to("cuda")


    lens = Lens()
    processor = LensProcessor()
    lendataset = LensDataset(ds=dataset, processor=processor)
    print(len(lendataset))  

    dataloader = DataLoader(lendataset, batch_size=args.batch_size, shuffle=False)

    # 检查是否有中断文件和索引
    output_file = f'bli2_{args.dataset_name}_tags_attributes.txt'
    start_idx = 0
    # if os.path.exists(output_file + '.idx'):
    #     with open(output_file + '.idx', 'r') as f_idx:
    #         start_idx = int(f_idx.read().strip())

    # dataloader = DataLoader(lendataset, batch_size=args.batch_size, shuffle=False)

    # 从中断的地方开始，或者从头开始
    generate_tags_attributes_for_batches(dataloader, lens, output_file, start_idx=start_idx)
    # generate_tags_attributes_for_one_batch(dataloader, lens)