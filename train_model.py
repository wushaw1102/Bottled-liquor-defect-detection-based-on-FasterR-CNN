import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# 数据集类
class WhiteLiquorDataset(Dataset):
    def __init__(self, img_folder, annotations_file, transform=None, is_train=True):
        # 初始化数据集
        self.img_folder = img_folder  # 图片文件夹路径
        self.transform = transform  # 数据转换操作
        self.is_train = is_train  # 是否为训练集

        # 读取注释文件
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)

        # 根据图片id构建图片信息字典
        self.images_info = {img['id']: img for img in self.annotations['images']}
        self.annotations_by_image = {}

        # 将注释按图片id分类
        for ann in self.annotations['annotations']:
            self.annotations_by_image.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        # 返回数据集长度
        return len(self.images_info)

    def __getitem__(self, idx):
        # 获取指定索引的样本
        image_info = list(self.images_info.values())[idx]  # 获取图片信息
        image_path = os.path.join(self.img_folder, image_info['file_name'])  # 构建图片路径
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # 读取并转换图片颜色通道

        if self.transform:
            image = self.transform(image)  # 如果有转换操作，应用它
        else:
            # 如果没有转换，将图像转换为张量并进行归一化
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        annotations = self.annotations_by_image.get(image_info['id'], [])  # 获取该图片的注释
        boxes = [
            [x, y, x + w, y + h] for ann in annotations for x, y, w, h in [ann['bbox']]
        ]
        labels = [ann['category_id'] for ann in annotations]  # 获取标签

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),  # 将边界框转换为张量
            'labels': torch.tensor(labels, dtype=torch.int64)  # 将标签转换为张量
        }

        return image, target  # 返回图像和目标

# 数据集划分
def split_dataset(annotations_file, train_ratio=0.7, val_ratio=0.2):
    # 划分数据集为训练集、验证集和测试集
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    image_ids = [img['id'] for img in data['images']]  # 获取所有图片id
    random.shuffle(image_ids)  # 随机打乱图片id

    total = len(image_ids)
    train_end = int(total * train_ratio)  # 训练集结束索引
    val_end = train_end + int(total * val_ratio)  # 验证集结束索引

    datasets = {'train': [], 'val': [], 'test': []}
    for img in data['images']:
        # 根据划分结果将图片分类到相应的集合中
        target = ('train' if img['id'] in image_ids[:train_end] else
                  'val' if img['id'] in image_ids[train_end:val_end] else 'test')
        datasets[target].append(img)

    # 将划分后的数据集保存为JSON文件
    for split, split_data in datasets.items():
        output_path = f'./data/{split}_annotations.json'
        with open(output_path, 'w') as f:
            json.dump({"images": split_data, "annotations": data['annotations'], "categories": data['categories']}, f)

# 数据增强
class IdentityTransform:
    # 作为一个身份转换，当不需要任何增强时用作占位符
    def __call__(self, x):
        return x

def get_transform(train=True):
    # 根据是否为训练集构建数据增强操作
    transforms_list = [
        transforms.ToPILImage(),  # 转换为PIL图像
        transforms.RandomHorizontalFlip(0.5) if train else IdentityTransform(),  # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2) if train else IdentityTransform(),  # 随机调整颜色
        transforms.RandomRotation(10) if train else IdentityTransform(),  # 随机旋转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ]
    return transforms.Compose(transforms_list)

# 模型类
class WhiteLiquorDetector:
    def __init__(self, num_classes=11):
        # 初始化检测器
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择设备
        self.model = self.get_model(num_classes).to(self.device)  # 初始化模型
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)  # 优化器
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)  # 学习率调度器

    def get_model(self, num_classes):
        # 获取Faster R-CNN模型
        model = fasterrcnn_resnet50_fpn(weights='DEFAULT')  # 默认权重加载
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # 修改分类器以适配类别数
        return model

    def train_one_epoch(self, train_loader, print_interval=50):
        # 训练一个epoch
        self.model.train()  # 设置为训练模式
        total_loss = 0  # 总损失初始化

        for step, (images, targets) in enumerate(train_loader, start=1):
            images = [img.to(self.device) for img in images]  # 移动图像到设备
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]  # 移动目标到设备

            # 计算损失
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            self.optimizer.zero_grad()  # 清空梯度
            losses.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
            total_loss += losses.item()  # 累加损失

            if step % print_interval == 0:
                print(f"Step {step}: Average Loss = {total_loss / step:.4f}")  # 打印损失信息

        return total_loss / len(train_loader)  # 返回平均损失

    def validate(self, val_loader):
        # 验证模型
        self.model.eval()  # 设置为评估模式
        with torch.no_grad():  # 关闭梯度计算
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]  # 移动图像到设备
                self.model(images)  # 前向传播
        print("Validation completed.")

    def train(self, train_loader, val_loader, epochs=4):
        # 训练过程
        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)  # 训练一个epoch
            self.validate(val_loader)  # 验证
            self.scheduler.step()  # 更新学习率
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

    def save_model(self, filename="model.pth"):
        # 保存模型
        torch.save(self.model.state_dict(), filename)  # 保存模型状态
        print(f"Model saved: {filename}")

# 主程序
if __name__ == '__main__':
    split_dataset('./data/annotations.json')  # 划分数据集
    train_dataset = WhiteLiquorDataset('./data/images', './data/train_annotations.json', transform=get_transform(train=True))  # 创建训练数据集
    val_dataset = WhiteLiquorDataset('./data/images', './data/val_annotations.json', transform=get_transform(train=False), is_train=False)  # 创建验证数据集
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))  # 创建训练数据加载器
    val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False, num_workers=8, collate_fn=lambda x: tuple(zip(*x)))  # 创建验证数据加载器

    detector = WhiteLiquorDetector(num_classes=11)  # 初始化检测器实例
    detector.train(train_loader, val_loader, epochs=10)  # 训练检测器
    detector.save_model()  # 保存模型