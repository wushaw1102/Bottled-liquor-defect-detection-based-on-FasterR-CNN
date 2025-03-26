import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

class WhiteLiquorDataset(Dataset):
    def __init__(self, img_folder, annotations_file, transform=None, is_train=True):
        self.img_folder = img_folder
        self.transform = transform
        self.is_train = is_train

        # 加载标注文件
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # 构建图像信息映射
        self.images_info = {img['id']: img for img in self.annotations['images']}
        self.annotations_by_image = {}

        for ann in self.annotations['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image:
                self.annotations_by_image[image_id] = []
            self.annotations_by_image[image_id].append(ann)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        # 获取图像信息
        image_info = list(self.images_info.values())[idx]
        image_path = os.path.join(self.img_folder, image_info['file_name'])

        # 读取图像
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image at path {image_path} is invalid.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            # 如果图像读取失败，用空白图像替代
            image = np.zeros((224, 224, 3), dtype=np.uint8)

        # 数据增强（仅在训练时）
        if self.is_train and self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # 获取该图像的标注
        image_id = image_info['id']
        annotations = self.annotations_by_image.get(image_id, [])

        # 如果没有标签，返回空标签
        if len(annotations) == 0:
            print(f"Warning: Image {image_info['file_name']} has no annotations.")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = []
            labels = []
            for ann in annotations:
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        return image, {'boxes': boxes, 'labels': labels}

# class WhiteLiquorDataset(Dataset):
#     def __init__(self, img_folder, annotations_file, transform=None, is_train=True):
#         self.img_folder = img_folder
#         self.transform = transform
#         self.is_train = is_train
#
#         # 加载标注文件
#         with open(annotations_file, 'r') as f:
#             self.annotations = json.load(f)
#
#         # 构建图像信息映射
#         self.images_info = {img['id']: img for img in self.annotations['images']}
#         self.annotations_by_image = {}
#
#         for ann in self.annotations['annotations']:
#             image_id = ann['image_id']
#             if image_id not in self.annotations_by_image:
#                 self.annotations_by_image[image_id] = []
#             self.annotations_by_image[image_id].append(ann)
#
#     def __len__(self):
#         return len(self.images_info)
#
#     def __getitem__(self, idx):
#         # 获取图像信息
#         image_info = list(self.images_info.values())[idx]
#         image_path = os.path.join(self.img_folder, image_info['file_name'])
#
#         # 读取图像
#         image = cv2.imread(image_path)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # 数据增强（仅在训练时）
#         if self.is_train and self.transform:
#             image = self.transform(image)
#         else:
#             # 确保总是转换为张量
#             image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#
#         # 获取该图像的标注
#         image_id = image_info['id']
#         annotations = self.annotations_by_image.get(image_id, [])
#
#         # 处理标注
#         boxes = []
#         labels = []
#         for ann in annotations:
#             x, y, w, h = ann['bbox']
#             boxes.append([x, y, x + w, y + h])
#             labels.append(ann['category_id'])
#
#         # 转换为张量
#         boxes = torch.tensor(boxes, dtype=torch.float32)
#         labels = torch.tensor(labels, dtype=torch.int64)
#
#         return image, {'boxes': boxes, 'labels': labels}


def split_dataset(annotations_file, train_ratio=0.7, val_ratio=0.2):
    """
    数据集划分
    :param annotations_file: 原始标注文件路径
    :param train_ratio: 训练集比例
    :param val_ratio: 验证集比例
    :return: 划分后的数据集标注
    """
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # 获取所有图像ID
    image_ids = [img['id'] for img in data['images']]
    random.shuffle(image_ids)

    # 计算划分索引
    total = len(image_ids)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 划分数据集
    train_ids = set(image_ids[:train_end])
    val_ids = set(image_ids[train_end:val_end])
    test_ids = set(image_ids[val_end:])

    # 创建新的数据集字典
    datasets = {
        'train': {'images': [], 'annotations': [], 'categories': data['categories']},
        'val': {'images': [], 'annotations': [], 'categories': data['categories']},
        'test': {'images': [], 'annotations': [], 'categories': data['categories']}
    }

    # 分配图像和标注
    for img in data['images']:
        if img['id'] in train_ids:
            datasets['train']['images'].append(img)
        elif img['id'] in val_ids:
            datasets['val']['images'].append(img)
        else:
            datasets['test']['images'].append(img)

    for ann in data['annotations']:
        if ann['image_id'] in train_ids:
            datasets['train']['annotations'].append(ann)
        elif ann['image_id'] in val_ids:
            datasets['val']['annotations'].append(ann)
        else:
            datasets['test']['annotations'].append(ann)

    # 保存划分后的数据集
    for split, split_data in datasets.items():
        output_path = f'./data/{split}_annotations.json'
        with open(output_path, 'w') as f:
            json.dump(split_data, f)

    return datasets


# 数据增强
def get_transform(train=True):
    transforms_list = []
    if train:
        # 训练时的数据增强
        transforms_list.extend([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
    else:
        transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)


# 使用示例
if __name__ == '__main__':
    # 划分数据集
    split_dataset('./data/annotations.json')

    # 创建数据加载器
    train_dataset = WhiteLiquorDataset(
        img_folder='./data/images',
        annotations_file='./data/train_annotations.json',
        transform=get_transform(train=True)
    )

    val_dataset = WhiteLiquorDataset(
        img_folder='./data/images',
        annotations_file='./data/val_annotations.json',
        transform=get_transform(train=False),
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)


