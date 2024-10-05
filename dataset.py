import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import List, Dict, Tuple
import random
import logging
import sys

# 在文件开头添加以下代码来配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FewShotDataset(Dataset):
    def __init__(self, root_dir: str):
        """
        Args:
            root_dir: 数据集根目录
            transform: 图像变换
            is_train: 是否为训练集
        """
        self.root_dir = root_dir

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),  # 添加随机水平翻转
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 获取所有类别
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 获取所有图片路径和标签
        self.images = []
        self.labels = []
        for cls_name in self.classes:
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    def sample_episode(
        self, n_way: int, k_shot: int, q_query: int
    ) -> Dict[str, torch.Tensor]:
        """采样一个训练episode"""
        # 随机选择n_way个类别
        selected_classes = random.sample(self.classes, n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        # 为每个类别采样支持集和查询集
        for label, cls_name in enumerate(selected_classes):
            cls_path = os.path.join(self.root_dir, cls_name)
            cls_images = os.listdir(cls_path)

            # 采样K-shot支持集
            selected_support = random.sample(cls_images, k_shot)
            for img_name in selected_support:
                img_path = os.path.join(cls_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                support_images.append(img)
                support_labels.append(label)

            # 采样查询集
            remaining_images = list(set(cls_images) - set(selected_support))
            selected_query = random.sample(remaining_images, q_query)
            for img_name in selected_query:
                img_path = os.path.join(cls_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                query_images.append(img)
                query_labels.append(label)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels),
            "query_images": torch.stack(query_images),
            "query_labels": torch.tensor(query_labels),
        }


class TestDataset:
    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: 测试集根目录，包含多个task文件夹
        """
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.task_dirs = [
            d for d in sorted(os.listdir(root_dir)) if d.startswith("task")
        ]

    def load_task(self, task_idx: int) -> Dict[str, torch.Tensor]:
        """加载指定task的数据"""
        task_dir = os.path.join(self.root_dir, self.task_dirs[task_idx])
        support_dir = os.path.join(task_dir, "support")
        query_dir = os.path.join(task_dir, "query")

        # 为当前任务创建class_to_idx
        class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(sorted(os.listdir(support_dir)))
        }

        # 加载支持集
        support_images = []
        support_labels = []
        for cls_name in sorted(os.listdir(support_dir)):
            cls_path = os.path.join(support_dir, cls_name)
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                support_images.append(img)
                support_labels.append(class_to_idx[cls_name])

        # 加载查询集
        query_images = []
        query_paths = []  # 保存查询集图片路径，用于结果提交
        for img_name in sorted(os.listdir(query_dir)):
            img_path = os.path.join(query_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            query_images.append(img)
            query_paths.append(img_path)

        return {
            "support_images": torch.stack(support_images),
            "support_labels": torch.tensor(support_labels),
            "query_images": torch.stack(query_images),
            "query_paths": query_paths,
            "class_to_idx": class_to_idx
        }


# 数据加载示例
def get_dataloaders(
    train_dir: str, test_dir: str, val_size: float, is_train: bool, batch_size: int = 32
):
    if is_train:
        # 训练集数据加载器
        train_dataset = FewShotDataset(train_dir)

        # 划分训练集和验证集
        train_size = int(len(train_dataset) * (1 - val_size))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=8
        )

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
        )
        logger.info(
            "train size: {}, val size: {}.".format(
                train_size, val_size
            )
        )
        return train_loader, val_loader
    else:
        # 测试集数据加载器

        test_dataset = TestDataset(test_dir)

        logger.info(
            "test task num: {}".format(
                 len(test_dataset.task_dirs)
            )
        )
        return test_dataset


if __name__ == "__main__":
    train_dir = r"./dataset/train_set"
    test_dir = r"./dataset/test_set"
    train_loader, val_loader, test_dataset = get_dataloaders(train_dir, test_dir,is_train=True)
