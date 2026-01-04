"""
装备模型信息管理模块
"""

import os
import csv
from typing import Dict, List, Optional
from collections import defaultdict


_gear_model_data: Optional[Dict[str, Dict]] = None
_model_groups: Optional[Dict[str, List[str]]] = None


def load_gear_model_info(csv_path: str = None):
    """
    加载装备模型信息CSV文件
    
    Args:
        csv_path: CSV文件路径，如果为None则从环境变量或默认路径读取
    """
    global _gear_model_data, _model_groups
    
    if csv_path is None:
        csv_path = os.getenv('GEAR_MODEL_INFO_CSV', 'data/gear_model_info.csv')
    
    if not os.path.exists(csv_path):
        print(f"[GearModel] Warning: CSV file not found: {csv_path}")
        _gear_model_data = {}
        _model_groups = defaultdict(list)
        return
    
    _gear_model_data = {}
    _model_groups = defaultdict(list)
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                item_id = row.get('物品ID', '')
                item_name = row.get('物品名称', '').strip('"')
                model_path = row.get('模型路径', '')
                
                if not item_name or not model_path:
                    continue
                
                _gear_model_data[item_id] = {
                    'id': item_id,
                    'name': item_name,
                    'model_path': model_path
                }
                
                _model_groups[model_path].append(item_id)
        
        print(f"[GearModel] Loaded {len(_gear_model_data)} gear items, {len(_model_groups)} model groups")
    except Exception as e:
        print(f"[GearModel] Error loading CSV file: {e}")
        _gear_model_data = {}
        _model_groups = defaultdict(list)


def get_gear_info(gear_id: str) -> Optional[Dict]:
    """
    获取装备信息
    
    Args:
        gear_id: 装备ID
        
    Returns:
        装备信息字典，包含id, name, model_path
    """
    if _gear_model_data is None:
        load_gear_model_info()
    
    return _gear_model_data.get(gear_id)


def get_same_model_gears(gear_label: str) -> List[Dict[str, str]]:
    """
    获取同模型的其他装备
    
    Args:
        gear_label: 装备label（格式："装备名称_物品ID"）
        
    Returns:
        同模型装备列表，每个元素包含id和name（不包括自身）
    """
    if _gear_model_data is None or _model_groups is None:
        load_gear_model_info()
    
    if not _gear_model_data:
        return []
    
    if '_' not in gear_label:
        return []
    
    gear_id = gear_label.rsplit('_', 1)[1]
    gear_info = get_gear_info(gear_id)
    if not gear_info:
        return []
    
    model_path = gear_info.get('model_path')
    if not model_path:
        return []
    
    same_model_gear_ids = _model_groups.get(model_path, [])
    
    same_model_gears = []
    for gid in same_model_gear_ids:
        if gid != gear_id:
            gear_info_item = _gear_model_data.get(gid)
            if gear_info_item:
                same_model_gears.append({
                    'id': gear_info_item['id'],
                    'name': gear_info_item['name']
                })
    
    return same_model_gears


def get_gear_model_info():
    """
    获取装备模型信息数据（用于调试）
    
    Returns:
        装备模型信息字典
    """
    if _gear_model_data is None:
        load_gear_model_info()
    
    return _gear_model_data

