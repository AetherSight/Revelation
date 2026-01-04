"""
数据库模型和操作
"""

import os
from datetime import datetime
from typing import Optional, List
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

Base = declarative_base()


class FeedbackRecord(Base):
    """反馈记录模型"""
    __tablename__ = "feedback_records"
    
    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(512), nullable=False, comment="图片存储路径或URL")
    label = Column(String(256), nullable=False, comment="正确的装备label")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "image_path": self.image_path,
            "label": self.label,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


_db_path = os.getenv('FEEDBACK_DB_PATH', 'data/feedback.db')
_engine = None
_SessionLocal = None


def init_db():
    """初始化数据库"""
    global _engine, _SessionLocal
    
    db_path = _db_path
    if not db_path.startswith('/') and not db_path.startswith('sqlite:///'):
        db_path = f"sqlite:///{db_path}"
    elif not db_path.startswith('sqlite:///'):
        db_path = f"sqlite:///{db_path}"
    
    _engine = create_engine(
        db_path,
        connect_args={"check_same_thread": False} if "sqlite" in db_path else {}
    )
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    Base.metadata.create_all(bind=_engine)


def get_db() -> Session:
    """获取数据库会话"""
    if _SessionLocal is None:
        init_db()
    
    db = _SessionLocal()
    try:
        return db
    finally:
        pass


def close_db(db: Session):
    """关闭数据库会话"""
    if db:
        db.close()


def create_feedback_record(image_path: str, label: str) -> FeedbackRecord:
    """创建反馈记录"""
    db = get_db()
    try:
        record = FeedbackRecord(
            image_path=image_path,
            label=label
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record
    except Exception as e:
        db.rollback()
        raise
    finally:
        close_db(db)


def get_feedback_records(skip: int = 0, limit: int = 100) -> List[FeedbackRecord]:
    """获取反馈记录列表"""
    db = get_db()
    try:
        records = db.query(FeedbackRecord).offset(skip).limit(limit).all()
        return records
    finally:
        close_db(db)


def get_feedback_record_by_id(record_id: int) -> Optional[FeedbackRecord]:
    """根据ID获取反馈记录"""
    db = get_db()
    try:
        record = db.query(FeedbackRecord).filter(FeedbackRecord.id == record_id).first()
        return record
    finally:
        close_db(db)

