"""
API路由定义
"""

import asyncio
from fastapi import File, UploadFile, Form, HTTPException, Query

from .schemas import HealthResponse, PredictionResponse, FeedbackResponse, AutocompleteResponse
from ..ml.loader import get_model, get_gallery, load_model
from ..ml.predictor import predict_image
from ..data.storage import get_storage_backend
from ..data.database import init_db, create_feedback_record
from ..data.gear_model import load_gear_model_info, search_gears_by_name, autocomplete_gear_names, get_same_model_gears


def setup_routes(app):
    """设置路由"""
    
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health():
        """健康检查"""
        model = get_model()
        gallery_embs, _ = get_gallery()
        
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "gallery_loaded": gallery_embs is not None
        }
    
    @app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
    async def predict(
        image: UploadFile = File(..., description="Image file to predict")
    ):
        """预测接口 - 通过文件上传（支持并发），固定返回top-10结果"""
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        top_k = 10
        image_data = await image.read()
        result = await asyncio.to_thread(predict_image, image_data, top_k)
        
        if len(result["results"]) > top_k:
            result["results"] = result["results"][:top_k]
        
        return result
    
    @app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
    async def feedback(
        image: UploadFile = File(..., description="用户标记的图片区域"),
        label: str = Form(..., description="正确的装备label")
    ):
        """反馈接口 - 接收用户标记的图片区域和正确的装备label"""
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        if not label or not label.strip():
            raise HTTPException(status_code=400, detail="label cannot be empty")
        
        try:
            image_data = await image.read()
            storage = get_storage_backend()
            image_path = await storage.save(image_data, image.filename or "image.jpg")
            create_feedback_record(image_path=image_path, label=label.strip())
            
            return {
                "status": "success"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
    
    @app.get("/search/autocomplete", response_model=AutocompleteResponse, tags=["Search"])
    async def autocomplete(
        q: str = Query(..., description="Search query for gear name autocomplete"),
        limit: int = Query(10, ge=1, le=50, description="Maximum number of suggestions")
    ):
        """装备名称自动补全接口"""
        if not q or not q.strip():
            return {"suggestions": []}
        
        suggestions = autocomplete_gear_names(q.strip(), limit)
        return {"suggestions": suggestions}
    
    @app.get("/search", response_model=PredictionResponse, tags=["Search"])
    async def search(
        q: str = Query(..., description="Search query for gear name"),
        limit: int = Query(10, ge=1, le=50, description="Maximum number of results")
    ):
        """装备名称搜索接口 - 返回格式与预测接口相同，包含同模装备信息"""
        if not q or not q.strip():
            return {"results": []}
        
        search_results = search_gears_by_name(q.strip(), limit)
        
        results = []
        for i, gear in enumerate(search_results, 1):
            label = gear['label']
            same_model_gears = get_same_model_gears(label)
            results.append({
                "rank": i,
                "label": label,
                "score": 1.0,  # 搜索结果的score设为1.0
                "same_model_gears": same_model_gears
            })
        
        return {"results": results}
    
    @app.on_event("startup")
    async def startup_event():
        """启动时加载模型和gallery"""
        import os
        
        try:
            init_db()
            print("[Startup] ✓ Database initialized")
        except Exception as e:
            print(f"[Startup] ⚠ Database initialization warning: {e}")
        
        try:
            storage = get_storage_backend()
            storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
            print(f"[Startup] ✓ Storage backend initialized: {storage_type}")
        except Exception as e:
            print(f"[Startup] ⚠ Storage backend initialization warning: {e}")
        
        try:
            load_gear_model_info()
            print("[Startup] ✓ Gear model info loaded")
        except Exception as e:
            print(f"[Startup] ⚠ Gear model info loading warning: {e}")
        
        model = get_model()
        gallery_embs, _ = get_gallery()
        
        if model is None or gallery_embs is None:
            print("[Startup] Loading model and gallery...")
            await asyncio.to_thread(load_model)
            
            model = get_model()
            gallery_embs, gallery_labels = get_gallery()
            
            if model is None:
                raise RuntimeError("Failed to load model during startup")
            if gallery_embs is None or gallery_labels is None:
                raise RuntimeError("Failed to load gallery during startup")
            
            print("[Startup] ✓ Model and gallery loaded successfully!")
        
        print("Server ready!")

