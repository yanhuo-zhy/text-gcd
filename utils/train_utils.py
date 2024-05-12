import os
import logging
import numpy as np
import torch
import random
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
from scipy.optimize import linear_sum_assignment as linear_assignment

from torch.nn import functional as F
from tqdm import tqdm
from faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from faster_mix_k_means_pytorch import pairwise_distance

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_experiment_directory(experiment_name, output_dir):
    # current_time = datetime.now().strftime('%m-%d-%H-%M')
    # experiment_dir = os.path.join(output_dir, f"{current_time}-{experiment_name}")
    # os.makedirs(experiment_dir, exist_ok=True)
    experiment_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    
    subdirs = ['logs', 'models', 'tensorboard_logs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
        
    return experiment_dir

def validate_evaluation_path(evaluate_path):
    essential_paths = [
        evaluate_path,
        os.path.join(evaluate_path, 'logs', 'log.txt'),
        os.path.join(evaluate_path, 'models', 'model.pth')
    ]
    for path in essential_paths:
        if not os.path.exists(path):
            raise ValueError(f"Invalid path: {path}. Please provide a valid path for evaluation using --evaluate_path")

def configure_logger(log_path):
    logging.basicConfig(filename=log_path,
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

# def modify_args_based_on_dataset(args):
#     if args.dataset_name == 'cub':
#         args.train_classes = range(100)
#         args.num_labeled_classes = len(args.train_classes)
#         args.unlabeled_classes = range(100, 200)
#         args.num_unlabeled_classes = len(args.unlabeled_classes)
#         args.interpolation = 3
#         args.crop_pct = 0.875
#         args.image_size = 224

# def modify_args_based_on_config(args):
#     if args.config_path:
#         config = configparser.ConfigParser()
#         config.read(args.config)
#         for key, value in config['DEFAULT'].items():
#             if key in args:
#                 if isinstance(args.key, int):
#                     setattr(args, key, config.getint('DEFAULT', key))
#                 elif isinstance(args.key, float):
#                     setattr(args, key, config.getfloat('DEFAULT', key))
#                 else:
#                     setattr(args, key, value)

def configure_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # if device == "cuda":
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = True
    return device

def log_initial_info(logger, args):
    action = "Training" if not args.evaluate else "Evaluating"
    logger.info(f"{action} {args.experiment_name} with the following settings:")
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    logger.info(f'Command-line arguments: {args_str}')

def init_tensorboard(tensorboard_log_dir, args):
    writer = SummaryWriter(tensorboard_log_dir)
    args_str = '\n '.join(f'{k}={v}' for k, v in vars(args).items())
    writer.add_text('Arguments', args_str)
    return writer


def init_experiment(args):
    # Setting Seeds
    set_seeds(args.seed_num)
    
    # Directory Management
    if not args.evaluate:
        if hasattr(args, 'interrupted_path') and os.path.exists(args.interrupted_path):
            experiment_dir = args.interrupted_path
        else:
            experiment_dir = create_experiment_directory(args.experiment_name, args.output_dir)
        
        args.log_path = os.path.join(experiment_dir, 'logs', 'log.txt')
        args.model_path = os.path.join(experiment_dir, 'models', 'model.pth')
        tensorboard_log_dir = os.path.join(experiment_dir, 'tensorboard_logs')

    else:
        validate_evaluation_path(args.evaluate_path)
        args.log_path = os.path.join(args.evaluate_path, 'logs', 'log.txt')
        args.model_path = os.path.join(args.evaluate_path, 'models', 'model.pth')
    
    # Logger Configuration
    logger = configure_logger(args.log_path)
    
    # Modify args based on dataset
    # modify_args_based_on_dataset(args)
    # modify_args_based_on_config(args)  
    
    # Device Configuration
    args.device = configure_device()
    
    # Logging Initial Info
    log_initial_info(logger, args)
    
    # TensorBoard Initialization
    if not args.evaluate:
        writer = init_tensorboard(tensorboard_log_dir, args)
        return args, logger, writer

    return args, logger

def get_pseudolabel(model, dataloader, pseudo_num):

    logits_image_list = []
    logits_text_list = []
    image_id_list = []

    for images, _, image_id, tag_token, _ in dataloader:
        image_id_list.append(image_id)
        images = images.cuda(non_blocking=True)
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)

        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, tag_token)
        
        logits_image_list.append(logits_image)
        logits_text_list.append(logits_text)

    logits_image_all = torch.cat(logits_image_list, dim=0)
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

    #----------------------------------------------------
    logits_image_all = F.softmax(logit_scale * logits_image_all, dim=-1)
    print("logits_image.shape", logits_image_all.shape)
    top_k_per_images = [logits_image_all[:, i].argsort(descending=True)[:pseudo_num] for i in range(logits_image_all.shape[1])]

    # 创建字典
    image_to_class_map_i = {}
    # image_id_flattened = [img_id for sublist in image_id_list for img_id in sublist]  # 展平列表
    for idx, image_indices in enumerate(top_k_per_images):
        # if idx >= old_class:
        for image_index in image_indices:
            image_id = image_id_flattened[image_index]
            image_to_class_map_i[image_id] = idx

    return image_to_class_map, image_to_class_map_i

def get_pseudolabel_twoimage(model, dataloader, pseudo_num):

    logits_image_list = []
    logits_text_list = []
    image_id_list = []

    for images, _, image_id, _, _ in dataloader:
        image_id_list.append(image_id)
        images = images.cuda(non_blocking=True)
        # tag_token = tag_token.squeeze(1).cuda(non_blocking=True)

        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, images)
        
        logits_image_list.append(logits_image)
        logits_text_list.append(logits_text)

    logits_image_all = torch.cat(logits_image_list, dim=0)
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

    #----------------------------------------------------
    logits_image_all = F.softmax(logit_scale * logits_image_all, dim=-1)
    print("logits_image.shape", logits_image_all.shape)
    top_k_per_images = [logits_image_all[:, i].argsort(descending=True)[:pseudo_num] for i in range(logits_image_all.shape[1])]

    # 创建字典
    image_to_class_map_i = {}
    # image_id_flattened = [img_id for sublist in image_id_list for img_id in sublist]  # 展平列表
    for idx, image_indices in enumerate(top_k_per_images):
        # if idx >= old_class:
        for image_index in image_indices:
            image_id = image_id_flattened[image_index]
            image_to_class_map_i[image_id] = idx

    return image_to_class_map, image_to_class_map_i

def get_pseudolabel_twotext(model, dataloader, pseudo_num):

    logits_image_list = []
    logits_text_list = []
    image_id_list = []

    for _, _, image_id, tag_token, _ in dataloader:
        image_id_list.append(image_id)
        # images = images.cuda(non_blocking=True)
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)

        with torch.no_grad():
            logits_image, logits_text, _, _ = model(tag_token, tag_token)
        
        logits_image_list.append(logits_image)
        logits_text_list.append(logits_text)

    logits_image_all = torch.cat(logits_image_list, dim=0)
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

    #----------------------------------------------------
    logits_image_all = F.softmax(logit_scale * logits_image_all, dim=-1)
    print("logits_image.shape", logits_image_all.shape)
    top_k_per_images = [logits_image_all[:, i].argsort(descending=True)[:pseudo_num] for i in range(logits_image_all.shape[1])]

    # 创建字典
    image_to_class_map_i = {}
    # image_id_flattened = [img_id for sublist in image_id_list for img_id in sublist]  # 展平列表
    for idx, image_indices in enumerate(top_k_per_images):
        # if idx >= old_class:
        for image_index in image_indices:
            image_id = image_id_flattened[image_index]
            image_to_class_map_i[image_id] = idx

    return image_to_class_map, image_to_class_map_i

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

def evaluate_two(model, test_loader, train_classes=None):
    model.eval()

    pred_text, pred_image, targets = [], [], []
    mask = np.array([])
    for _, (images, label, _, tag_token, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, tag_token)

            pred_text.append(logits_text.argmax(1).cpu().numpy())
            pred_image.append(logits_image.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    pred_text = np.concatenate(pred_text)
    pred_image = np.concatenate(pred_image)
    targets = np.concatenate(targets)
    
    # 预测精度-text
    total_acc_text, old_acc_text, new_acc_text = evaluate_accuracy(pred_text, targets, mask)

    # 预测精度-image
    total_acc_image, old_acc_image, new_acc_image = evaluate_accuracy(pred_image, targets, mask)

    return total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image

def evaluate_two_images(model, test_loader, train_classes=None):
    model.eval()

    pred_text, pred_image, targets = [], [], []
    mask = np.array([])
    for _, (images, label, _, tag_token, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        # tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, images)

            pred_text.append(logits_text.argmax(1).cpu().numpy())
            pred_image.append(logits_image.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    pred_text = np.concatenate(pred_text)
    pred_image = np.concatenate(pred_image)
    targets = np.concatenate(targets)
    
    # 预测精度-text
    total_acc_text, old_acc_text, new_acc_text = evaluate_accuracy(pred_text, targets, mask)

    # 预测精度-image
    total_acc_image, old_acc_image, new_acc_image = evaluate_accuracy(pred_image, targets, mask)

    return total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image

def evaluate_two_text(model, test_loader, train_classes=None):
    model.eval()

    pred_text, pred_image, targets = [], [], []
    mask = np.array([])
    for _, (_, label, _, tag_token, _) in enumerate(tqdm(test_loader)):
        # images = images.cuda(non_blocking=True)
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(tag_token, tag_token)

            pred_text.append(logits_text.argmax(1).cpu().numpy())
            pred_image.append(logits_image.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    pred_text = np.concatenate(pred_text)
    pred_image = np.concatenate(pred_image)
    targets = np.concatenate(targets)
    
    # 预测精度-text
    total_acc_text, old_acc_text, new_acc_text = evaluate_accuracy(pred_text, targets, mask)

    # 预测精度-image
    total_acc_image, old_acc_image, new_acc_image = evaluate_accuracy(pred_image, targets, mask)

    return total_acc_text, old_acc_text, new_acc_text, total_acc_image, old_acc_image, new_acc_image

def evaluate_weighted(model, test_loader, train_classes=None):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _, tag_token, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, tag_token)

            classifier_image_probs = F.softmax(logits_image, dim=-1)
            classifier_text_probs = F.softmax(logits_text, dim=-1)

            averaged_probs = 0.5 * classifier_image_probs + 0.5 * classifier_text_probs

            preds.append(averaged_probs.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 预测精度
    total_acc, old_acc, new_acc = evaluate_accuracy(preds, targets, mask)

    return total_acc, old_acc, new_acc

def evaluate_weighted_twoimage(model, test_loader, train_classes=None):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        # tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(images, images)

            classifier_image_probs = F.softmax(logits_image, dim=-1)
            classifier_text_probs = F.softmax(logits_text, dim=-1)

            averaged_probs = 0.5 * classifier_image_probs + 0.5 * classifier_text_probs

            preds.append(averaged_probs.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 预测精度
    total_acc, old_acc, new_acc = evaluate_accuracy(preds, targets, mask)

    return total_acc, old_acc, new_acc

def evaluate_weighted_twotext(model, test_loader, train_classes=None):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (_, label, _, tag_token, _) in enumerate(tqdm(test_loader)):
        # images = images.cuda(non_blocking=True)
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():
            logits_image, logits_text, _, _ = model(tag_token, tag_token)

            classifier_image_probs = F.softmax(logits_image, dim=-1)
            classifier_text_probs = F.softmax(logits_text, dim=-1)

            averaged_probs = 0.5 * classifier_image_probs + 0.5 * classifier_text_probs

            preds.append(averaged_probs.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # 预测精度
    total_acc, old_acc, new_acc = evaluate_accuracy(preds, targets, mask)

    return total_acc, old_acc, new_acc

def kmeans_semi_sup(args, merge_test_loader, model, K=None, train_classes=None):

    """
    In this case, the test loader needs to have the labelled and unlabelled subsets of the training data
    """

    if K is None:
        raise ValueError("K should not be None.")

    all_feats = []
    targets = np.array([])
    mask_lab = np.array([])     # From all the data, which instances belong to the labelled set
    mask_cls = np.array([])     # From all the data, which instances belong to Old classes
    image_id_list = []

    print('Collating features...')
    # First extract all features
    for batch_idx, (_, label, image_id, tag_token, _, mask_lab_) in enumerate(tqdm(merge_test_loader)):
        image_id_list.append(image_id)

        mask_lab_ = mask_lab_[:, 0]
        label, mask_lab_ = label.cuda(non_blocking=True), mask_lab_.cuda(non_blocking=True).bool()
        tag_token = tag_token.squeeze(1).cuda(non_blocking=True)
        with torch.no_grad():

            feats = model.encode_text(tag_token)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in train_classes else False for x in label]))
        mask_lab = np.append(mask_lab, mask_lab_.cpu().bool().numpy())

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_lab = mask_lab.astype(bool)
    mask_cls = mask_cls.astype(bool)

    all_feats = np.concatenate(all_feats)
    # pca = PCA(n_components=50)  # 降维到50维
    # all_feats = pca.fit_transform(all_feats)

    l_feats = all_feats[mask_lab]       # Get labelled set
    u_feats = all_feats[~mask_lab]      # Get unlabelled set
    l_targets = targets[mask_lab]       # Get labelled targets
    u_targets = targets[~mask_lab]       # Get unlabelled targets

    print('Fitting Semi-Supervised K-Means...')
    kmeans = SemiSupKMeans(k=K, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                           n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

    l_feats, u_feats, l_targets, u_targets = (torch.from_numpy(x).to('cuda') for
                                              x in (l_feats, u_feats, l_targets, u_targets))

    kmeans.fit_mix(u_feats, l_feats, l_targets)
    all_preds = kmeans.labels_.cpu().numpy()
    kmeans_centers = kmeans.cluster_centers_
    return kmeans_centers

def kmeans_centers_evaluate(model, test_loader, kmeans_centers, train_classes=None):
    model.eval()

    preds, targets = [], []
    mask = np.array([])
    for batch_idx, (images, label, _, _, _) in enumerate(tqdm(test_loader)):
        images = images.cuda(non_blocking=True)
        with torch.no_grad():
            _, _, image_features = model(images)

            logits = 100.0 * image_features @ kmeans_centers.T
            preds.append(logits.argmax(1).cpu().numpy())
            targets.append(label.cpu().numpy())
            mask = np.append(mask, np.array([True if x.item() in train_classes else False for x in label]))

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    
    # 预测精度
    total_acc, old_acc, new_acc = evaluate_accuracy(preds, targets, mask)

    return total_acc, old_acc, new_acc

def introduce_typo(word):
    if len(word) < 3:
        return word
    
    # 选定要修改的字符的位置
    char_idx = random.randint(1, len(word) - 2)
    
    # 以一定的概率选择替换、删除或添加
    action = random.choice(['replace', 'delete', 'add'])
    
    if action == 'replace':
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        word = word[:char_idx] + random_char + word[char_idx + 1:]
    elif action == 'delete':
        word = word[:char_idx] + word[char_idx + 1:]
    else:  # add
        random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
        word = word[:char_idx] + random_char + word[char_idx:]
    
    return word

def tokenize_with_augmentation(text):
    # 使用空格进行分词
    words = text.split()
    
    # 以某种概率对词进行拼写错误增强
    augmented_words = [introduce_typo(word) if random.random() < 0.1 else word for word in words]
    
    return ' '.join(augmented_words)