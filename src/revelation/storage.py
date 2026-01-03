"""
存储抽象层 - 支持本地文件系统和腾讯云COS
"""

import os
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from datetime import datetime


class StorageBackend(ABC):
    """存储后端抽象基类"""
    
    @abstractmethod
    async def save(self, image_data: bytes, filename: str) -> str:
        """
        保存图片数据
        
        Args:
            image_data: 图片二进制数据
            filename: 文件名
            
        Returns:
            存储路径或URL
        """
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        删除文件
        
        Args:
            path: 存储路径或URL
            
        Returns:
            是否删除成功
        """
        pass


class LocalFileStorage(StorageBackend):
    """本地文件系统存储"""
    
    def __init__(self, base_dir: str = "feedback_images"):
        """
        初始化本地文件存储
        
        Args:
            base_dir: 存储基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        today = datetime.now().strftime("%Y-%m-%d")
        self.today_dir = self.base_dir / today
        self.today_dir.mkdir(parents=True, exist_ok=True)
    
    async def save(self, image_data: bytes, filename: str) -> str:
        """保存图片到本地文件系统"""
        file_ext = Path(filename).suffix or ".jpg"
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        file_path = self.today_dir / unique_filename
        file_path.write_bytes(image_data)
        return str(file_path.relative_to(self.base_dir))
    
    async def delete(self, path: str) -> bool:
        """删除本地文件"""
        try:
            file_path = self.base_dir / path
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception:
            return False
    
    def get_full_path(self, relative_path: str) -> str:
        """获取文件的完整路径"""
        return str(self.base_dir / relative_path)


class TencentCOSStorage(StorageBackend):
    """腾讯云COS存储"""
    
    def __init__(
        self,
        secret_id: str,
        secret_key: str,
        region: str,
        bucket: str,
        base_path: str = "feedback"
    ):
        """
        初始化腾讯云COS存储
        
        Args:
            secret_id: 腾讯云SecretId
            secret_key: 腾讯云SecretKey
            region: COS地域
            bucket: 存储桶名称
            base_path: COS中的基础路径
        """
        try:
            from qcloud_cos import CosConfig
            from qcloud_cos import CosS3Client
        except ImportError:
            raise ImportError(
                "qcloud_cos not installed. Install it with: pip install cos-python-sdk-v5"
            )
        
        self.secret_id = secret_id
        self.secret_key = secret_key
        self.region = region
        self.bucket = bucket
        self.base_path = base_path.rstrip('/')
        
        config = CosConfig(
            Region=region,
            SecretId=secret_id,
            SecretKey=secret_key
        )
        self.client = CosS3Client(config)
    
    async def save(self, image_data: bytes, filename: str) -> str:
        """上传图片到腾讯云COS"""
        import asyncio
        
        file_ext = Path(filename).suffix or ".jpg"
        today = datetime.now().strftime("%Y/%m/%d")
        unique_filename = f"{uuid.uuid4().hex}{file_ext}"
        cos_key = f"{self.base_path}/{today}/{unique_filename}"
        
        def _upload():
            self.client.put_object(
                Bucket=self.bucket,
                Body=image_data,
                Key=cos_key,
                ContentType="image/jpeg"
            )
            return cos_key
        
        await asyncio.to_thread(_upload)
        return f"cos://{self.bucket}/{cos_key}"
    
    async def delete(self, path: str) -> bool:
        """删除COS文件"""
        import asyncio
        
        try:
            if path.startswith("cos://"):
                parts = path.replace("cos://", "").split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    if bucket != self.bucket:
                        return False
                else:
                    key = parts[0]
            else:
                key = path
            
            def _delete():
                self.client.delete_object(
                    Bucket=self.bucket,
                    Key=key
                )
            
            await asyncio.to_thread(_delete)
            return True
        except Exception:
            return False


def get_storage_backend() -> StorageBackend:
    """
    根据环境变量获取存储后端实例
    
    环境变量:
        STORAGE_TYPE: 存储类型，'local' 或 'cos' (默认: 'local')
        FEEDBACK_STORAGE_DIR: 本地存储目录 (默认: 'feedback_images')
        COS_SECRET_ID: 腾讯云SecretId
        COS_SECRET_KEY: 腾讯云SecretKey
        COS_REGION: COS地域
        COS_BUCKET: COS存储桶名称
        COS_BASE_PATH: COS基础路径 (默认: 'feedback')
    """
    storage_type = os.getenv('STORAGE_TYPE', 'local').lower()
    
    if storage_type == 'cos':
        secret_id = os.getenv('COS_SECRET_ID')
        secret_key = os.getenv('COS_SECRET_KEY')
        region = os.getenv('COS_REGION')
        bucket = os.getenv('COS_BUCKET')
        base_path = os.getenv('COS_BASE_PATH', 'feedback')
        
        if not all([secret_id, secret_key, region, bucket]):
            raise ValueError(
                "COS storage requires COS_SECRET_ID, COS_SECRET_KEY, "
                "COS_REGION, and COS_BUCKET environment variables"
            )
        
        return TencentCOSStorage(
            secret_id=secret_id,
            secret_key=secret_key,
            region=region,
            bucket=bucket,
            base_path=base_path
        )
    else:
        base_dir = os.getenv('FEEDBACK_STORAGE_DIR', 'feedback_images')
        return LocalFileStorage(base_dir=base_dir)

