"""
预测模块
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from fastapi import HTTPException

from .dataset import imread_unicode
from .model_loader import get_model, get_gallery, get_transform, get_device


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
        # 读取图片
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = imread_unicode(image_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 预处理
        query = transform(img).unsqueeze(0).to(device)
        query_emb = model(query).cpu()
        query_emb = F.normalize(query_emb, dim=1)

        # 计算相似度
        sims = torch.matmul(query_emb, gallery_embs.T)[0]  # [N]

        # 计算所有相似度并排序
        all_idxs = torch.argsort(sims, descending=True)
        
        # 收集 top_k 个唯一标签（按相似度从高到低）
        seen = {}  # label -> best score
        for idx in all_idxs.tolist():
            label = gallery_labels[idx]
            if label not in seen:
                seen[label] = sims[idx].item()
                # 当收集到足够的唯一标签时停止
                if len(seen) >= top_k:
                    break

        # 最终Top-K：按分数排序并取前 top_k 个
        final = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 格式化结果
        results = []
        for i, (label, score) in enumerate(final, 1):
            results.append({
                "rank": i,
                "label": label,
                "score": float(score)
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

