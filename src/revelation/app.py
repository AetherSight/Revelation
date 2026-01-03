"""
FastAPI 应用主文件
"""

import os
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from .schemas import HealthResponse, PredictionResponse, FeedbackResponse
from .model_loader import load_model, get_model, get_gallery
from .predictor import predict_image
from .storage import get_storage_backend
from .database import init_db, create_feedback_record

app = FastAPI(
    title="Revelation",
    description="FFXIV equipment recognition microservice by AetherSight",
    version="0.1.0"
)


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


@app.on_event("startup")
async def startup_event():
    """启动时加载模型和gallery"""
    import asyncio
    
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


def main():
    """主函数 - 启动服务"""
    import uvicorn
    
    print("=" * 50)
    print("Starting Revelation service...")
    print("=" * 50)
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'true').lower() == 'true'
    
    if debug:
        print(f"Starting web server in DEBUG mode on port {port}...")
        print("Model will be loaded during application startup.")
        print("=" * 50)
        uvicorn.run(
            "revelation.app:app",
            host="0.0.0.0", 
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        print("\n[1/2] Loading model and gallery...")
        try:
            load_model()
            print("[1/2] ✓ Model and gallery loaded successfully!")
        except Exception as e:
            print(f"[1/2] ✗ Error loading model: {e}")
            raise
        
        model = get_model()
        gallery_embs, gallery_labels = get_gallery()
        
        if model is None:
            raise RuntimeError("Model is None after loading")
        if gallery_embs is None or gallery_labels is None:
            raise RuntimeError("Gallery is None after loading")
        
        print(f"[2/2] Starting web server on port {port}...")
        print("=" * 50)
        uvicorn.run(
            app,
            host="0.0.0.0", 
            port=port,
            reload=False,
            log_level="info"
        )


if __name__ == '__main__':
    main()
