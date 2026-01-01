"""
FastAPI 应用主文件
"""

import os
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
    """预测接口 - 通过文件上传"""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_data = await image.read()
    result = predict_image(image_data, top_k=top_k)
    return result


@app.on_event("startup")
async def startup_event():
    """启动时检查模型是否已加载"""
    model = get_model()
    gallery_embs, _ = get_gallery()
    
    if model is None:
        raise RuntimeError("Model not loaded before server startup")
    if gallery_embs is None:
        raise RuntimeError("Gallery not loaded before server startup")
    
    print("Server ready!")


def main():
    """主函数 - 先加载模型，再启动服务"""
    import uvicorn
    
    print("=" * 50)
    print("Starting Revelation service...")
    print("=" * 50)
    
    # 同步加载模型和gallery（阻塞直到完成）
    print("\n[1/2] Loading model and gallery...")
    try:
        load_model()
        print("[1/2] ✓ Model and gallery loaded successfully!")
    except Exception as e:
        print(f"[1/2] ✗ Error loading model: {e}")
        raise
    
    # 验证加载状态
    model = get_model()
    gallery_embs, gallery_labels = get_gallery()
    
    if model is None:
        raise RuntimeError("Model is None after loading")
    if gallery_embs is None or gallery_labels is None:
        raise RuntimeError("Gallery is None after loading")
    
    print(f"[2/2] Starting web server on port {os.getenv('PORT', 5000)}...")
    print("=" * 50)
    
    # 启动服务
    port = int(os.getenv('PORT', 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == '__main__':
    main()
