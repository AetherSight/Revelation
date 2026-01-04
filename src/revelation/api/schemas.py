"""
Pydantic 模型定义
"""

from pydantic import BaseModel
from typing import List, Dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gallery_loaded: bool


class SameModelGear(BaseModel):
    id: str
    name: str


class PredictionResult(BaseModel):
    rank: int
    label: str
    score: float
    same_model_gears: List[SameModelGear] = []


class PredictionResponse(BaseModel):
    results: List[PredictionResult]


class FeedbackResponse(BaseModel):
    status: str


class AutocompleteResponse(BaseModel):
    suggestions: List[str]