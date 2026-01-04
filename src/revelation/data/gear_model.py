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
                
                if not item_id or not item_name or not model_path:
                    continue
                
                _gear_model_data[item_id] = {
                    'id': item_id,
                    'name': item_name,
                    'model_path': model_path
                }
                
                _model_groups[model_path].append(item_id)
    except Exception as e:
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


def search_gears_by_name(query: str, limit: int = 10) -> List[Dict]:
    """
    根据装备名称搜索装备
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        匹配的装备列表，每个元素包含id, name, label（格式："装备名称_物品ID"）
    """
    if _gear_model_data is None:
        load_gear_model_info()
    
    if not _gear_model_data or not query:
        return []
    
    query = query.strip()
    if not query:
        return []
    
    results = []
    query_lower = query.lower()
    
    for gear_id, gear_info in _gear_model_data.items():
        gear_name = gear_info.get('name', '')
        if query_lower in gear_name.lower():
            label = f"{gear_name}_{gear_id}"
            results.append({
                'id': gear_id,
                'name': gear_name,
                'label': label
            })
    
    # 按名称长度和匹配位置排序（精确匹配优先）
    results.sort(key=lambda x: (
        x['name'].lower().startswith(query_lower),
        x['name'].lower().index(query_lower) if query_lower in x['name'].lower() else 999,
        len(x['name'])
    ))
    
    return results[:limit]


def autocomplete_gear_names(query: str, limit: int = 10) -> List[str]:
    """
    装备名称自动补全
    
    Args:
        query: 搜索关键词
        limit: 返回结果数量限制
        
    Returns:
        匹配的装备名称列表
    """
    if _gear_model_data is None:
        load_gear_model_info()
    
    if not _gear_model_data or not query:
        return []
    
    query = query.strip()
    if not query:
        return []
    
    names = set()
    query_lower = query.lower()
    
    for gear_info in _gear_model_data.values():
        gear_name = gear_info.get('name', '')
        if query_lower in gear_name.lower():
            names.add(gear_name)
    
    # 按名称长度和匹配位置排序
    sorted_names = sorted(names, key=lambda x: (
        x.lower().startswith(query_lower),
        x.lower().index(query_lower) if query_lower in x.lower() else 999,
        len(x)
    ))
    
    return sorted_names[:limit]


def get_gear_model_info():
    """
    获取装备模型信息数据（用于调试）
    
    Returns:
        装备模型信息字典
    """
    if _gear_model_data is None:
        load_gear_model_info()
    
    return _gear_model_data

