"""
预测模块
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from fastapi import HTTPException

from .dataset import imread_unicode
from .loader import get_model, get_gallery, get_transform, get_device
from ..data.gear_model import get_same_model_gears


def predict_image(image_data, top_k=5):
    """
    对图片进行预测
    
    Args:
        image_data: 图片数据（bytes或文件路径）
        top_k: 返回Top-K结果
    
    Returns:
        预测结果字典
    """
    model = get_model()
    gallery_embs, gallery_labels = get_gallery()
    transform = get_transform()
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

        query = transform(img).unsqueeze(0).to(device)
        query_emb = model(query).cpu()
        query_emb = F.normalize(query_emb, dim=1)

        sims = torch.matmul(query_emb, gallery_embs.T)[0]

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

