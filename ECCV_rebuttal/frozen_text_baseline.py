import sys
sys.path.append("..")

import os
from scipy.io import loadmat
from torchvision.datasets import CIFAR100
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from torch.optim import SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from data.get_datasets import get_datasets, get_class_splits
from model import CustomCLIP, CustomCosineAnnealingLR, ImageViewGenerator, TextViewGenerator, load_clip_to_cpu
from data.augmentations import get_transform
from utils import init_experiment, get_pseudolabel, evaluate_two, evaluate_weighted

from clip import clip
from scipy.optimize import linear_sum_assignment as linear_assignment

def evaluate_accuracy(preds, targets, mask):
    # 预测精度
    mask = mask.astype(bool)
    targets = targets.astype(int)
    preds = preds.astype(int)

    old_classes_gt = set(targets[mask])
    new_classes_gt = set(targets[~mask])

    assert preds.size == targets.size
    D = max(preds.max(), targets.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(preds.size):
        w[preds[i], targets[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    # ind_match = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind])
    total_instances = preds.size

    total_acc /= total_instances

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc

def evaluate_two_frozentext(model, test_loader, train_classes=None):
    model.eval()

    pred_text, pred_image, pred_all, targets = [], [], [], []
    mask = np.array([])
    for _, (images, label, _, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            logits_image, logits_text, _ = model(images)

            classifier_image_probs = F.softmax(logits_image, dim=-1)
            classifier_text_probs = F.softmax(logits_text, dim=-1)

            averaged_probs = 0.5 * classifier_image_probs + 0.5 * classifier_text_probs

            pred_text.append(logits_text.argmax(1).cpu().numpy())
            pred_image.append(logits_image.argmax(1).cpu().numpy())
            pred_all.append(averaged_probs.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    pred_text = np.concatenate(pred_text)
    pred_image = np.concatenate(pred_image)
    pred_all = np.concatenate(pred_all)
    targets = np.concatenate(targets)
    
    # 预测精度-text
    total_acc_text, old_acc_text, new_acc_text = evaluate_accuracy(pred_text, targets, mask)

    # 预测精度-image
    total_acc_image, old_acc_image, new_acc_image = evaluate_accuracy(pred_image, targets, mask)

    # 预测精度-all
    total_acc_all, old_acc_all, new_acc_all = evaluate_accuracy(pred_all, targets, mask)

    return total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image, total_acc_all, old_acc_all, new_acc_all

def get_pseudolabel_frozentext(model, dataloader, pseudo_num):

    logits_text_list = []
    image_id_list = []

    for images, _, image_id, _, _ in dataloader:
        image_id_list.append(image_id)
        images = images.cuda(non_blocking=True)

        with torch.no_grad():
            _, logits_text, _ = model(images)
        
        logits_text_list.append(logits_text)

    logits_text_all = torch.cat(logits_text_list, dim=0)
    logit_scale = 100.0

    #---------------------------------------------------
    logits_text_all = F.softmax(logit_scale * logits_text_all, dim=-1)
    print("logits_text.shape", logits_text_all.shape)

    top_k_per_text = [logits_text_all[:, i].argsort(descending=True)[:pseudo_num] for i in range(logits_text_all.shape[1])]

    # 创建字典
    image_to_class_map = {}
    image_id_flattened = [img_id for sublist in image_id_list for img_id in sublist]  # 展平列表
    for idx, image_indices in enumerate(top_k_per_text):

        for image_index in image_indices:
            image_id = image_id_flattened[image_index]
            image_to_class_map[image_id] = idx

    return image_to_class_map


class FrozenTextCLIP(nn.Module):
    def __init__(self, clip_model, class_nums, dataset_name):
        super().__init__()
        self.model = clip_model
        self.outputdim = clip_model.visual.output_dim

        self.image_classifier = nn.utils.weight_norm(nn.Linear(self.outputdim, class_nums, bias=False))
        self.image_classifier.weight_g.data.fill_(1)
        self.image_classifier.weight_g.requires_grad = False

        if dataset_name == 'cub':
            cub_root = "/db/pszzz/NCD_dataset/cub"
            with open(os.path.join(cub_root, 'CUB_200_2011', 'classes.txt'), 'r') as f:
                lines = f.readlines()
                class_descriptions = [line.strip().split('.', 1)[1] for line in lines]
            self.class_descriptions = [desc.replace('_', ' ') for desc in class_descriptions]
            
        elif dataset_name == 'scars':
            file_path = '/db/pszzz/NCD_dataset/stanford_cars/cars_annos.mat'
            data = loadmat(file_path)
            class_names_array = data['class_names'][0]
            self.class_descriptions = [str(name[0]) for name in class_names_array]

        elif dataset_name == 'cifar100':
            cifar100_root = "/db/pszzz/NCD_dataset/cifar100"
            data = CIFAR100(root=cifar100_root, train=True)
            self.class_descriptions = data.classes

        else:
            pass

        self.class_descriptions = ["A photo of a " + desc for desc in self.class_descriptions]
        text_tokens = clip.tokenize(self.class_descriptions)
        with torch.no_grad():
            self.frozen_classname = self.model.encode_text(text_tokens)
            self.frozen_classname = self.frozen_classname / self.frozen_classname.norm(dim=-1, keepdim=True)
        self.frozen_classname = self.frozen_classname.to('cuda')
        print("frozen_classname.shape", self.frozen_classname.shape)

    def encode_image(self, image):
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def forward(self, images):
        image_features = self.encode_image(images)

        logit_scale = self.model.logit_scale.exp()
        logits_text = logit_scale * image_features @ self.frozen_classname.t()

        logits_image = self.image_classifier(image_features)

        return logits_image, logits_text, image_features

def train_one_epoch(args, logger, writer, loader, model, optimizer, scheduler, epoch, image_to_class_map):
    model.train()
    total_loss = 0.0
    total_loss_cls = 0.0
    total_loss_cluster = 0.0
    total_loss_pseduo = 0.0
    total_loss_clip_tag = 0.0


    teacher_temp_schedule = np.concatenate((np.linspace(0.035,0.02, 30),np.ones(200 - 30) * 0.02))
    param_group_names = ['classifier_head', 'base_parameters']

    for batch_idx, (images, labels, img_id, _, _, mask) in enumerate(tqdm(loader, desc="Training")):
        mask = mask[:, 0]
        labels, mask = labels.cuda(non_blocking=True), mask.cuda(non_blocking=True).bool()
        images = torch.cat(images, dim=0).cuda(non_blocking=True)
        # tag_token = torch.cat(tag_token, dim=0).squeeze(1).cuda(non_blocking=True)

        logits_image, logits_text, image_features = model(images)

        #-------------------------------------------------------Clip loss--------------------------------------------------------#

        # # cosine similarity as logits
        # logit_scale = model.model.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features_tag.t()
        # ground truth
        # ground_truth = torch.arange(len(image_features),dtype=torch.long, device=images.device)
        # Calculate the loss
        # loss_clip_tag = F.cross_entropy(logits_per_image, ground_truth)
        logits_per_image = torch.cat([f[mask] for f in logits_text.chunk(2)], dim=0)
        ground_truth = torch.cat([labels[mask] for _ in range(2)], dim=0)        
        loss_clip_tag = nn.CrossEntropyLoss()(logits_per_image, ground_truth)

        #-------------------------------------------------------SimGCD cls loss--------------------------------------------------#

        sup_logits = torch.cat([f[mask] for f in (logits_image / 0.1).chunk(2)], dim=0)
        sup_labels = torch.cat([labels[mask] for _ in range(2)], dim=0)
        loss_cls = nn.CrossEntropyLoss()(sup_logits, sup_labels)

        # sup_logits_text = torch.cat([f[mask] for f in (logits_text / 0.1).chunk(2)], dim=0)
        # loss_cls += nn.CrossEntropyLoss()(sup_logits_text, sup_labels)        

        #-----------------------------------------------------SimGCD cluster loss------------------------------------------------#

        student_out = logits_image / 0.05
        teacher_out = logits_image.detach()
        student_out = student_out.chunk(2)

        temp = teacher_temp_schedule[epoch]
        teacher_out = F.softmax(teacher_out / temp, dim=-1)
        teacher_out = teacher_out.chunk(2)
        loss_cluster = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss_cluster += torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1).mean()
                n_loss_terms += 1

        loss_cluster /= n_loss_terms

        avg_probs = (logits_image / 0.05).softmax(dim=1).mean(dim=0)
        me_max_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
        loss_cluster += 2 * me_max_loss
        #--------------------------------------------------------------------
        student_out_text = logits_text
        teacher_out_text = logits_text.detach()
        student_out_text = student_out_text.chunk(2)

        teacher_temp_schedule_text = np.concatenate((np.linspace(0.7,0.4, 30),np.ones(200 - 30) * 0.4))
        temp = teacher_temp_schedule_text[epoch]
        teacher_out_text = F.softmax(teacher_out_text / temp, dim=-1)
        teacher_out_text = teacher_out_text.chunk(2)
        loss_cluster_text = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out_text):
            for v in range(len(student_out_text)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss_cluster_text += torch.sum(-q * F.log_softmax(student_out_text[v], dim=-1), dim=-1).mean()
                n_loss_terms += 1

        loss_cluster_text /= n_loss_terms

        avg_probs_text = logits_text.softmax(dim=1).mean(dim=0)
        me_max_loss_text = - torch.sum(torch.log(avg_probs_text**(-avg_probs_text))) + math.log(float(len(avg_probs_text)))
        loss_cluster_text += 2 * me_max_loss_text

        loss_cluster += loss_cluster_text

        #----------------------------------------------------------pseudo loss--------------------------------------------------#

        pseudo_labels = []
        selected_logits = []

        if image_to_class_map:
            for idx, id in enumerate(img_id+img_id):
                if id in image_to_class_map:
                    pseudo_label = image_to_class_map[id]
                    pseudo_labels.append(pseudo_label)
                    selected_logits.append(logits_image[idx])

        # 如果存在伪标签，计算交叉熵损失
        if pseudo_labels:
            pseudo_labels_tensor = torch.tensor(pseudo_labels).to(logits_image.device)  # 将列表转化为张量，并移到适当的设备上
            selected_logits_tensor = torch.stack(selected_logits)  # 将张量列表堆叠成一个新的张量
            selected_logits_tensor = selected_logits_tensor / 0.1

            criterion = nn.CrossEntropyLoss()
            loss_pseduo = criterion(selected_logits_tensor, pseudo_labels_tensor)
        else:
            loss_pseduo = torch.tensor(0.0,device=logits_image.device)
        # #-------------------------------------------
        # if image_to_class_map_i:
        #     pseudo_labels_i = []
        #     selected_logits_i = []

        #     for idx, id in enumerate(img_id+img_id):
        #         if id in image_to_class_map_i:
        #             pseudo_label = image_to_class_map_i[id]
        #             pseudo_labels_i.append(pseudo_label)
        #             selected_logits_i.append(logits_text[idx])

        #     # 如果存在伪标签，计算交叉熵损失
        #     if pseudo_labels_i:
        #         pseudo_labels_i_tensor = torch.tensor(pseudo_labels_i).to(logits_text.device)  # 将列表转化为张量，并移到适当的设备上
        #         selected_logits_i_tensor = torch.stack(selected_logits_i)  # 将张量列表堆叠成一个新的张量
        #         selected_logits_i_tensor = selected_logits_i_tensor / 0.1

        #         criterion = nn.CrossEntropyLoss()
        #         loss_pseduo_i = criterion(selected_logits_i_tensor, pseudo_labels_i_tensor)
        #     else:
        #         loss_pseduo_i = torch.tensor(0.0,device=logits_text.device)
        #     loss_pseduo += loss_pseduo_i

        #-------------------------------------------------------------total loss--------------------------------------------------#

        loss = 0
        loss += args.lambda_loss * loss_cls 
        loss += (1-args.lambda_loss) * loss_cluster
        loss += loss_pseduo
        loss += loss_clip_tag

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_cls += loss_cls.item()
        total_loss_cluster += loss_cluster.item()
        total_loss_pseduo += loss_pseduo.item()
        total_loss_clip_tag += loss_clip_tag.item()

        iter_idx = epoch * len(train_loader) + batch_idx
        # writer.add_scalars('Loss', {
        #     'loss_cls': loss_cls.item(),
        #     'loss_cluster': loss_cluster.item(),
        #     'loss_pseduo': loss_pseduo.item(),
        #     'loss_clip_tag': loss_clip_tag.item(),
        #     'total_loss': loss.item()
        # }, iter_idx)

    scheduler.step()
    logger.info(
    f"Epoch {epoch+1}/{args.epochs}, Total Loss: {total_loss / len(loader):.4f}, "
    f"Cls Loss: {total_loss_cls / len(loader):.4f}, Cluster Loss: {total_loss_cluster / len(loader):.4f}, "
    f"New Loss: {total_loss_pseduo / len(loader):.4f}, Clip tag Loss: {total_loss_clip_tag / len(loader):.4f}"
    )
    for idx, param_group in enumerate(optimizer.param_groups):
        logger.info(f"   Param Group: {param_group_names[idx]}, Learning Rate: {param_group['lr']:.4f}")


def linear_increase(start_num, end_num, epochs):
    return [round(start_num + (end_num - start_num) * i / (epochs - 1)) for i in range(epochs)]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--output_dir', type=str, default='exp')
    parser.add_argument('--experiment_name', type=str, default='scars_topk10_pseudo(10-15)')
    parser.add_argument('--seed_num', type=int, default=1)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--dataset_name', type=str, default='scars', help='options: cifar10, cifar100, imagenet_100, cub, scars, aircraft, herbarium_19, pets, flowers, food')
    parser.add_argument('--backbone_name', type=str, default='ViT-B/16', help="choose from 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'")

    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--base_lr', type=float, default=0.0005)
    parser.add_argument('--classifier_lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--prop_train_labels', type=float, default=0.5)

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--crop_pct', type=float, default=0.875)
    parser.add_argument('--interpolation', type=int, default=3) 
    parser.add_argument('--transform', type=str, default='imagenet') 

    parser.add_argument('--alpha_sr', type=float, default=0)
    parser.add_argument('--alpha_ri', type=float, default=0.05)
    parser.add_argument('--alpha_rs', type=float, default=0.05)
    parser.add_argument('--alpha_rd', type=float, default=0.05)

    parser.add_argument('--pseudo_ratio', type=float, default=0.3)
    parser.add_argument('--lambda_loss', type=float, default=0.3)
    parser.add_argument('--coteaching_epoch_t', type=int, default=10)
    parser.add_argument('--coteaching_epoch_i', type=int, default=15)

    parser.add_argument('--max_kmeans_iter', type=int, default=10)
    parser.add_argument('--k_means_init', type=int, default=20)    

    parser.add_argument('--interrupted_path', type=str, default='') 

    args = parser.parse_args()
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes
    args, logger, writer = init_experiment(args)

    logger.info(f"Loading CLIP (backbone: {args.backbone_name})")
    clip_model = load_clip_to_cpu(args.backbone_name).float()
    # model_name: str = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    # clip_model = open_clip.create_model_and_transforms(model_name)[0]

    logger.info("Building custom CLIP")
    # model = CustomCLIP(clip_model, args.num_classes).to(args.device)
    model = FrozenTextCLIP(clip_model, args.num_classes, args.dataset_name).to(args.device)

    logger.info("Turning off gradients in the image encoder")
    for name, param in model.named_parameters():
        param.requires_grad_(False)

    for name, param in model.named_parameters():
        if "visual.transformer.resblocks.11" in name:
            param.requires_grad_(True)
        if "visual.proj" in name:
            param.requires_grad_(True)
        if "image_classifier" in name:
            param.requires_grad_(True)

    params_names = [name for name, param in model.named_parameters() if param.requires_grad]
    logger.info("Parameters that require gradients: %s", params_names)

    classifier_params_train = [p for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]
    classifier_params_train_name = [name for name, p in model.named_parameters() if "classifier" in name and p.requires_grad]
    logger.info("Parameters in classifier with big lr: %s", classifier_params_train_name)

    # 获取其他层的参数
    other_params_train = [p for name, p in model.named_parameters() if "classifier" not in name and p.requires_grad]

    # 设定学习率
    classifier_lr = args.classifier_lr
    base_lr = args.base_lr  # 其他层的默认学习率

    # 创建优化器，为不同的参数组设置不同的学习率
    optimizer_train = SGD([
        {'params': classifier_params_train, 'lr': classifier_lr},
        {'params': other_params_train, 'lr': base_lr}
    ])

    scheduler_train = CustomCosineAnnealingLR(optimizer_train, classifier_params_train, T_max=args.epochs, eta_min=classifier_lr*1e-3)


    # dataset
    # transform: text-[text,text] image-[simgcd_transform, simgcd_transform]
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ImageViewGenerator(base_transform=train_transform, n_views=2)

    # 创建数据加载器
    train_dataset, _, test_dataset, _ = get_datasets(args.dataset_name, train_transform, test_transform, None, args)
 
    # 采样器
    label_len = len(train_dataset.labelled_dataset)
    unlabelled_len = len(train_dataset.unlabelled_dataset)
    sample_weights = [1 if i < label_len else label_len / unlabelled_len for i in range(len(train_dataset))]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,sampler=sampler, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_size=256, shuffle=False, pin_memory=False)

    logger.info(f"len of train dataset: {len(train_loader.dataset)}")
    logger.info(f"len of test dataset: {len(test_loader.dataset)}")
    
    pseudo_num = math.floor(len(test_loader.dataset) / args.num_classes * args.pseudo_ratio)
    logger.info(f"Pseudo Nums: {pseudo_num}")

    best_acc_w = 0.0

    start_epoch = 0
    # if os.path.exists(args.model_path):
    #     checkpoint = torch.load(args.model_path)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer_train.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1  # 加1是为了从下一个epoch开始
    #     if 'scheduler_state_dict' in checkpoint:
    #         scheduler_train.load_state_dict(checkpoint['scheduler_state_dict'])

    for epoch in range(start_epoch, args.epochs):
        # image_to_class_map, image_to_class_map_i = get_pseudolabel(model, test_loader, pseudo_num=pseudo_num)
        image_to_class_map = get_pseudolabel_frozentext(model, test_loader, pseudo_num=pseudo_num)
        logger.info(f"len of image_to_class_map: {len(image_to_class_map)}")

        pseudo_text = None
        if epoch > args.coteaching_epoch_t:
            pseudo_text = image_to_class_map

        if epoch == 0:
            # total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image = evaluate_two(model, test_loader, train_classes=args.train_classes)
            total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image, total_acc_all, old_acc_all, new_acc_all = evaluate_two_frozentext(model, test_loader, train_classes=args.train_classes)
            logger.info(f"Before Train Accuracies: Text {total_acc_text:.4f} | Old {old_acc_text:.4f} | New {new_acc_text:.4f}")
            logger.info(f"Before Train Accuracies: Image {total_acc_image:.4f} | Old {old_acc_image:.4f} | New {new_acc_image:.4f}")
            logger.info(f"Before Train Accuracies: All {total_acc_all:.4f} | Old {old_acc_all:.4f} | New {new_acc_all:.4f}")


        train_one_epoch(args, logger, writer, train_loader, model, optimizer_train, scheduler_train, epoch, pseudo_text)
        # total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image = evaluate_two(model, test_loader, train_classes=args.train_classes)
        total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image, total_acc_all, old_acc_all, new_acc_all = evaluate_two_frozentext(model, test_loader, train_classes=args.train_classes)
        logger.info(f"Text classifier Epoch {epoch} Train Accuracies: Text All {total_acc_text:.4f} | Old {old_acc_text:.4f} | New {new_acc_text:.4f}")
        logger.info(f"Image classifier Epoch {epoch} Train Accuracies: Image All {total_acc_image:.4f} | Old {old_acc_image:.4f} | New {new_acc_image:.4f}")
        logger.info(f"All classifier Epoch {epoch} Train Accuracies: All All {total_acc_all:.4f} | Old {old_acc_all:.4f} | New {new_acc_all:.4f}")

        # total_acc_w, old_acc_w, new_acc_w = evaluate_weighted(model, test_loader, train_classes=args.train_classes)
        # logger.info(f"Weighted Accuracies: All {total_acc_w:.4f} | Old {old_acc_w:.4f} | New {new_acc_w:.4f}")
        # writer.add_scalar('Accuracy/All', total_acc_w, epoch)
        # writer.add_scalar('Accuracy/Old', old_acc_w, epoch)
        # writer.add_scalar('Accuracy/New', new_acc_w, epoch)

        # if total_acc_w > best_acc_w:
        #     best_acc_w = total_acc_w
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer_train.state_dict(),
        #         'scheduler_state_dict': scheduler_train.state_dict()
        #     }
        #     torch.save(checkpoint, args.model_path)

    writer.close()