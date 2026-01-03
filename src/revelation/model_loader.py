"""
模型加载和管理模块
"""

import os
import torch

from .model import EmbeddingModel
from .preprocess import InferenceTransform
from .gallery import build_gallery

# 全局状态
model = None
gallery_embs = None
gallery_labels = None
transform = None
device = None

MODEL_DIR = os.getenv('MODEL_DIR', 'models')
GALLERY_ROOT = os.getenv('GALLERY_ROOT', None)


def get_model():
    """获取模型实例"""
    return model


def get_gallery():
    """获取gallery数据"""
    return gallery_embs, gallery_labels


def get_transform():
    """获取transform"""
    return transform


def get_device():
    """获取设备"""
    return device


def load_model():
    """加载模型和gallery"""
    global model, transform, device, gallery_embs, gallery_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载主模型
    model_path = os.path.join(MODEL_DIR, "aethersight.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model = EmbeddingModel().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 初始化transform
    transform = InferenceTransform()

    # 加载gallery（优先使用缓存文件）
    gallery_cache_path = os.path.join(MODEL_DIR, "aethersight_gallery.pth")
    
    # 如果缓存文件存在，直接加载
    if os.path.exists(gallery_cache_path):
        print(f"[Gallery] Loading cache from {gallery_cache_path}...")
        data = torch.load(gallery_cache_path, map_location="cpu")
        gallery_embs = data["embs"]
        gallery_labels = data["labels"]
        print(f"[Gallery] Loaded {len(gallery_labels)} gallery items from cache")
    else:
        # 缓存不存在，需要从 GALLERY_ROOT 构建
        if not GALLERY_ROOT:
            raise ValueError(
                f"Gallery cache not found at {gallery_cache_path} and "
                "GALLERY_ROOT environment variable is not set. "
                "Either provide the cache file or set GALLERY_ROOT to build it."
            )
        
        if not os.path.exists(GALLERY_ROOT):
            raise FileNotFoundError(
                f"Gallery cache not found and GALLERY_ROOT directory does not exist: {GALLERY_ROOT}"
            )
        
        print(f"[Gallery] Cache not found, building gallery from {GALLERY_ROOT}...")
        gallery_embs, gallery_labels = build_gallery(
            model,
            GALLERY_ROOT,
            transform,
            device,
            batch_size=128,
            num_workers=8,
            cache_path=gallery_cache_path
        )
        
        if gallery_embs is None or gallery_labels is None:
            raise RuntimeError("Failed to build gallery embeddings")
        
        print(f"[Gallery] Built and saved {len(gallery_labels)} gallery items")

