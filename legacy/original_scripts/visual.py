import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')  # 使用非交互式后端
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import cv2
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# 加载模型
def load_model(model_path, num_classes=11):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)


# 数据集加载函数
def load_data(img_folder, annotations_file):
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)

    images_info = {img['id']: img for img in annotations['images']}
    annotations_by_image = {}
    for ann in annotations['annotations']:
        annotations_by_image.setdefault(ann['image_id'], []).append(ann)
    return images_info, annotations_by_image


# 可视化检测结果
def visualize_predictions(model, img_folder, images_info, annotations_by_image, transform):
    device = next(model.parameters()).device
    for image_id, image_info in images_info.items():
        image_path = os.path.join(img_folder, image_info['file_name'])
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        transformed_image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(transformed_image)[0]

        # 绘制结果
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        ax = plt.gca()

        # 显示真实标注
        for ann in annotations_by_image.get(image_id, []):
            x, y, w, h = ann['bbox']
            rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y, f"GT: {ann['category_id']}", color='green', fontsize=10)

        # 显示预测结果
        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score > 0.5:  # 阈值可调整
                x1, y1, x2, y2 = box.cpu().numpy()
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f"Pred: {label.item()} ({score:.2f})", color='red', fontsize=10)

        plt.title(f"Image ID: {image_id}")
        plt.savefig(f"./saved/result_{image_id}.png")
        plt.show()

        # plt.close()  # 关闭当前图像窗口

# # 混淆矩阵
# def plot_confusion_matrix(true_labels, pred_labels, class_names):
#     cm = confusion_matrix(true_labels, pred_labels, labels=range(len(class_names)))
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()


# # PR曲线
# def plot_pr_curve(true_labels, pred_probs, class_names):
#     plt.figure(figsize=(10, 8))
#     for i, class_name in enumerate(class_names):
#         precision, recall, _ = precision_recall_curve(true_labels == i, pred_probs[:, i])
#         ap = average_precision_score(true_labels == i, pred_probs[:, i])
#         plt.plot(recall, precision, label=f'{class_name} (AP={ap:.2f})')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend()
#     plt.savefig('pr_curve.png')
#     plt.show()



# 主函数
if __name__ == '__main__':
    model = load_model('model.pth', num_classes=11)
    img_folder = './data/images'
    annotations_file = './data/test_annotations.json'

    images_info, annotations_by_image = load_data(img_folder, annotations_file)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 可视化检测结果
    visualize_predictions(model, img_folder, images_info, annotations_by_image, transform)

    # 类别名称，按类别的 ID 顺序排列
    class_names = [
        '背景',
        '瓶盖破损',
        '瓶盖变形',
        '瓶盖坏边',
        '瓶盖打旋',
        '瓶盖断点',
        '标贴歪斜',
        '标贴起皱',
        '标贴气泡',
        '喷码正常',
        '喷码异常'
    ]

    # 主函数中的标签收集部分
    true_labels = []
    pred_labels = []
    pred_probs = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for image_id, image_info in images_info.items():
            image_path = os.path.join(img_folder, image_info['file_name'])
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            transformed_image = transform(image).unsqueeze(0).to(device)
            prediction = model(transformed_image)[0]

            # 收集真实标签
            for ann in annotations_by_image.get(image_id, []):
                if ann['category_id'] != 0:  # 排除背景类
                    true_labels.append(ann['category_id'])

            # 如果没有预测结果，添加一个默认值
            if len(prediction['scores']) == 0:
                pred_labels.append(0)  # 默认值表示没有检测到任何目标
                probs = torch.zeros(11)  # 对应11个类别
                pred_probs.append(probs.numpy())
            else:
                # 收集预测标签和概率
                for label, score in zip(prediction['labels'], prediction['scores']):
                    if score > 0.3:  # 试着降低阈值，比如改为0.3
                        pred_labels.append(label.item())
                        probs = torch.zeros(11)  # 对应11个类别
                        probs[label.item()] = score.item()
                        pred_probs.append(probs.numpy())


    # 绘制混淆矩阵
    # plot_confusion_matrix(true_labels, pred_labels, class_names)

    # 绘制PR曲线
    # plot_pr_curve(np.array(true_labels), pred_probs, class_names)

    # # 示例混淆矩阵和PR曲线可视化
    # true_labels = []
    # pred_labels = []
    # pred_probs = []
    #
    # device = next(model.parameters()).device
    # with torch.no_grad():
    #     for image_id, image_info in images_info.items():
    #         image_path = os.path.join(img_folder, image_info['file_name'])
    #         image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    #         transformed_image = transform(image).unsqueeze(0).to(device)
    #         prediction = model(transformed_image)[0]
    #
    #         for ann in annotations_by_image.get(image_id, []):
    #             true_labels.append(ann['category_id'])
    #
    #         for label, score in zip(prediction['labels'], prediction['scores']):
    #             pred_labels.append(label.item())
    #             probs = torch.zeros(11)
    #             probs[label.item()] = score.item()
    #             pred_probs.append(probs.numpy())
    #
    # pred_probs = np.array(pred_probs)
    #
    # # 绘制混淆矩阵
    # class_names = [f'Class {i}' for i in range(11)]
    # print(f"True labels length: {len(true_labels)}")
    # print(f"Pred labels length: {len(pred_labels)}")
    # print(f"Sample true labels: {true_labels[:5]}")
    # print(f"Sample pred labels: {pred_labels[:5]}")
    #
    # plot_confusion_matrix(true_labels, pred_labels, class_names)
    #
    # # 绘制PR曲线
    # plot_pr_curve(np.array(true_labels), pred_probs, class_names)
