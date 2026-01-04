"""
FastAPI 应用主文件
"""

import os
from fastapi import FastAPI

from .api.routes import setup_routes
from .ml.loader import load_model, get_model, get_gallery

app = FastAPI(
    title="Revelation",
    description="FFXIV equipment recognition microservice by AetherSight",
    version="0.1.0"
)

setup_routes(app)


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
