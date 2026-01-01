"""
数据集相关模块
"""

import numpy as np
import cv2
from torch.utils.data import Dataset


def imread_unicode(image_path):
    """读取支持中文路径的图片"""
    img_array = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


class GalleryDataset(Dataset):
    """
    用于构建 gallery embedding 的 Dataset
    """
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = imread_unicode(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img, self.labels[idx]

