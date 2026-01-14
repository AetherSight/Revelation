"""
图像预处理模块
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class PatchTransform:
    """
    Patch 图像变换（用于局部特征提取）
    将 patch 图像转换为 tensor
    """
    def __init__(self, return_tensor=True):
        self.return_tensor = return_tensor
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, image):
        """
        Args:
            image: numpy array (H, W, C) RGB格式或PIL Image
        
        Returns:
            torch.Tensor: [C, H, W] 归一化后的tensor
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        return self.transform(pil_image)


class InferenceTransform:
    """
    推理时的图像变换（只做resize、normalize和tensor转换）
    替代 A2ClothingTransform(train=False)
    """
    def __init__(self, size=512):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, image):
        """
        Args:
            image: numpy array (H, W, C) RGB格式
        
        Returns:
            torch.Tensor: [C, H, W] 归一化后的tensor
        """
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        return self.transform(pil_image)

