"""
模型相关模块
包含 EmbeddingModel
"""

import torch.nn as nn
import torch.nn.functional as F
import timm


class EmbeddingModel(nn.Module):
    """
    基于 EfficientNet 的嵌入模型
    用于生成归一化的特征向量
    支持全局和局部特征提取
    """
    def __init__(self, model_name="tf_efficientnetv2_m", emb_dim=512):
        """
        Args:
            model_name: timm 模型名称
            emb_dim: 嵌入维度
        """
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0  # 去掉分类头
        )
        self.global_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
        self.local_head = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
    
    def forward(self, x, return_local=False):
        """
        Args:
            x: 输入图像张量 [B, C, H, W]
            return_local: 是否返回局部特征
        
        Returns:
            如果 return_local=False: 返回 (global_emb, None)
            如果 return_local=True: 返回 (global_emb, local_emb)
            其中 emb 是归一化的嵌入向量 [B, emb_dim]
        """
        feat = self.backbone(x)
        global_emb = self.global_head(feat)
        global_emb = F.normalize(global_emb, dim=1)
        
        if return_local:
            local_emb = self.local_head(feat)
            local_emb = F.normalize(local_emb, dim=1)
            return global_emb, local_emb
        else:
            return global_emb, None

