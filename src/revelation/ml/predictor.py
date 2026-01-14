"""
预测模块
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from fastapi import HTTPException

from .dataset import imread_unicode
from .loader import get_model, get_gallery, get_transform, get_patch_transform, get_device
from ..data.gear_model import get_same_model_gears


def extract_patches_from_image(img, patch_transform, model, device, num_patches=5):
    """
    Extract patch features from image for evaluation.
    Uses fixed center-based patch positions for consistent results.
    """
    patch_size = 224
    H, W = img.shape[:2]
    
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    
    center_h = H / 2.0
    center_w = W / 2.0
    
    if num_patches == 1:
        offset_h_ratios = np.array([0.0])
        offset_w_ratios = np.array([0.0])
    elif num_patches == 5:
        offset_h_ratios = np.array([0.0, -0.15, 0.15, 0.0, 0.0])
        offset_w_ratios = np.array([0.0, 0.0, 0.0, -0.08, 0.08])
    else:
        n = num_patches
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        radius_h = 0.15
        radius_w = 0.08
        offset_h_ratios = radius_h * np.sin(angles)
        offset_w_ratios = radius_w * np.cos(angles)
    
    offset_h = offset_h_ratios * H
    offset_w = offset_w_ratios * W
    
    patch_center_h = center_h + offset_h
    patch_center_w = center_w + offset_w
    
    top_positions = (patch_center_h - patch_size / 2.0).astype(np.int32)
    left_positions = (patch_center_w - patch_size / 2.0).astype(np.int32)
    
    top_positions = np.clip(top_positions, 0, H - patch_size)
    left_positions = np.clip(left_positions, 0, W - patch_size)
    
    patches_list = []
    for p_idx in range(num_patches):
        top = top_positions[p_idx]
        left = left_positions[p_idx]
        
        patch = img[top:top+patch_size, left:left+patch_size]
        
        if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        
        patch_tensor = patch_transform(patch)
        patches_list.append(patch_tensor)
    
    patches = torch.stack(patches_list).to(device)
    _, patch_embs = model(patches, return_local=True)
    patch_embs = F.normalize(patch_embs, dim=1)
    
    del patches
    return patch_embs.cpu()


def predict_image(image_data, top_k=5, patch_weight=0.0, patch_only=False):
    """
    对图片进行预测
    
    Args:
        image_data: 图片数据（bytes或文件路径）
        top_k: 返回Top-K结果
        patch_weight: 局部patch权重（0-1浮点数），全局权重自动计算为 1 - patch_weight
        patch_only: 如果为True，将输入图像视为单个patch（不需要提取），图像本身用作patch特征进行匹配
    
    Returns:
        预测结果字典
    """
    if patch_only:
        global_weight = 0.0
        patch_weight = 1.0
    else:
        global_weight = 1.0 - patch_weight
    
    use_patch_match = patch_weight > 0
    num_patches = 5  # 从完整图像提取时的固定patch数量
    
    model = get_model()
    gallery_embs, gallery_labels = get_gallery()
    transform = get_transform()
    patch_transform = get_patch_transform() if use_patch_match else None
    device = get_device()

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if gallery_embs is None or gallery_labels is None:
        raise HTTPException(status_code=500, detail="Gallery not loaded")

    try:
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = imread_unicode(image_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        sims_global = None
        if not patch_only:
            query = transform(img).unsqueeze(0).to(device)
            query_emb, _ = model(query, return_local=False)
            query_emb = query_emb.cpu()
            query_emb = F.normalize(query_emb, dim=1)
            sims_global = torch.matmul(query_emb, gallery_embs.T)[0]
        
        sims_patch = None
        if use_patch_match:
            if patch_only:
                patch_tensor = patch_transform(img).unsqueeze(0).to(device)
                _, patch_emb = model(patch_tensor, return_local=True)
                patch_emb = F.normalize(patch_emb, dim=1).cpu()
                sims_patch = torch.matmul(patch_emb, gallery_embs.T)[0]
            else:
                patch_embs = extract_patches_from_image(
                    img, patch_transform, model, device, num_patches
                )
                
                sims_patch_list = []
                for patch_emb in patch_embs:
                    sims_patch = torch.matmul(patch_emb.unsqueeze(0), gallery_embs.T)[0]
                    sims_patch_list.append(sims_patch)
                
                sims_patch_stack = torch.stack(sims_patch_list)
                
                if num_patches > 2:
                    max_vals, _ = sims_patch_stack.max(dim=0)
                    min_vals, _ = sims_patch_stack.min(dim=0)
                    sims_patch = (sims_patch_stack.sum(dim=0) - max_vals - min_vals) / (num_patches - 2)
                else:
                    sims_patch = sims_patch_stack.mean(dim=0)
        
        if patch_only:
            sims = sims_patch
        elif use_patch_match:
            sims = global_weight * sims_global + patch_weight * sims_patch
        else:
            sims = sims_global

        all_idxs = torch.argsort(sims, descending=True)
        
        seen = {}
        for idx in all_idxs.tolist():
            label = gallery_labels[idx]
            if label not in seen:
                seen[label] = sims[idx].item()
                if len(seen) >= top_k:
                    break

        final = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for i, (label, score) in enumerate(final, 1):
            same_model_gears = get_same_model_gears(label)
            results.append({
                "rank": i,
                "label": label,
                "score": float(score),
                "same_model_gears": same_model_gears
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

