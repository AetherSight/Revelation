"""
Pydantic 模型定义
"""

from pydantic import BaseModel
from typing import List


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gallery_loaded: bool


class PredictionResult(BaseModel):
    rank: int
    label: str
    score: float


class PredictionResponse(BaseModel):
    results: List[PredictionResult]

