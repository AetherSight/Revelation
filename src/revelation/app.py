"""
FastAPI 应用主文件
"""

import os
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException

from .schemas import HealthResponse, PredictionResponse
from .model_loader import load_model, get_model, get_gallery
from .predictor import predict_image

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
    image: UploadFile = File(..., description="Image file to predict"),
    top_k: int = Form(5, description="Number of top results to return")
):
    """预测接口 - 通过文件上传（支持并发）"""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if top_k <= 0:
        top_k = 5
    
    image_data = await image.read()
    # 将同步的模型推理操作放到线程池中执行，避免阻塞事件循环
    # 这样多个请求可以并发处理，不会互相阻塞
    result = await asyncio.to_thread(predict_image, image_data, top_k)
    
    if len(result["results"]) > top_k:
        result["results"] = result["results"][:top_k]
    
    return result


@app.on_event("startup")
async def startup_event():
    """启动时加载模型和gallery"""
    import asyncio
    
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
