"""
Gallery 构建模块
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from .dataset import GalleryDataset


@torch.no_grad()
def build_gallery(
    model,
    gallery_root,
    transform,
    device,
    batch_size=128,
    cache_path=None,
    num_workers=8
):
    """
    构建gallery embeddings
    
    Args:
        model: 嵌入模型
        gallery_root: gallery图片根目录
        transform: 图像变换
        device: 设备
        batch_size: 批次大小
        cache_path: 缓存路径
        num_workers: DataLoader工作进程数
    
    Returns:
        gallery_embs: gallery嵌入向量
        gallery_labels: gallery标签列表
    """
    # 检查缓存
    if cache_path and os.path.exists(cache_path):
        print(f"[Gallery] Loading cache from {cache_path}")
        data = torch.load(cache_path, map_location="cpu")
        return data["embs"], data["labels"]

    model.eval()

    # 扫描gallery目录
    image_paths = []
    image_labels = []

    class_names = sorted(
        d.name for d in os.scandir(gallery_root) if d.is_dir()
    )

    for cls in class_names:
        cls_dir = os.path.join(gallery_root, cls)
        for name in os.listdir(cls_dir):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_paths.append(os.path.join(cls_dir, name))
                image_labels.append(cls)

    print(f"[Gallery] Total images: {len(image_paths)}")

    if len(image_paths) == 0:
        return None, None

    # Dataset / Loader
    dataset = GalleryDataset(
        image_paths=image_paths,
        labels=image_labels,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    # 推理
    gallery_embs_list = []
    gallery_labels_out = []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            emb = model(imgs)

        emb = F.normalize(emb, dim=1)

        gallery_embs_list.append(emb.cpu())
        gallery_labels_out.extend(labels)

    gallery_embs = torch.cat(gallery_embs_list, dim=0)

    # 保存缓存
    if cache_path:
        torch.save(
            {
                "embs": gallery_embs,
                "labels": gallery_labels_out,
            },
            cache_path
        )
        print(f"[Gallery] Cache saved to {cache_path}")

    return gallery_embs, gallery_labels_out

