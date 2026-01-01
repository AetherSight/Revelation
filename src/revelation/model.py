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
        self.head = nn.Sequential(
            nn.Linear(self.backbone.num_features, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )
    
    def forward(self, x):
        """
        Args:
            x: 输入图像张量 [B, C, H, W]
        
        Returns:
            归一化的嵌入向量 [B, emb_dim]
        """
        feat = self.backbone(x)
        emb = self.head(feat)
        emb = F.normalize(emb, dim=1)
        return emb

