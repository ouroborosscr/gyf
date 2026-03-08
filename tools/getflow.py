import json
import logging
import os
import sys
from datetime import datetime
from bson import ObjectId
from pymongo import MongoClient
from langchain.tools import tool
from typing import List, Dict, Any

# --- 1. 动态添加项目根目录到环境变量 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入配置
try:
    from utils import config
except ImportError:
    logging.warning("Warning: Could not import utils.config. Ensure project structure is correct.")
    config = None

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 我们仍然保留 Encoder 用于文件写入，但返回列表时我们手动处理类型
class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)

@tool
def export_flows_tool(
    skip: int = 0, 
    limit: int = 10, 
    output_filename: str = "dataset.json",
    conn_collection: str = "connections",
    payload_collection: str = "payloads",
    max_payload_len: int = 2000
) -> List[Dict[str, Any]]: # <--- 修改返回值类型提示
    """
    导出 Zeek 数据流。
    可以直接返回 Python 列表形式的数据，方便代码内部直接赋值和截取。
    """
    logging.info("正在使用 getflow 工具")
    
    # 1. 获取数据库连接
    try:
        mongo_cfg = config.DATABASE["mongo"]
        client = MongoClient(mongo_cfg["uri"])
        db = client[mongo_cfg["db_name"]]
    except Exception as e:
        logging.error(f"Database Connection Failed: {str(e)}")
        return [] # 出错返回空列表
        
    conn_col = db[conn_collection]
    payload_col = db[payload_collection]

    # 2. 查询 connections
    logging.info(f"正在查询 {conn_collection}: skip={skip}, limit={limit}...")
    try:
        cursor = conn_col.find({}).sort("ts", 1).skip(skip).limit(limit)
    except Exception as e:
        logging.error(f"Query Failed: {str(e)}")
        return []
    
    export_data = []
    processed_count = 0

    for conn_doc in cursor:
        uid = conn_doc.get("uid")
        if not uid:
            continue

        # 3. 聚合 Payload
        payload_cursor = payload_col.find({"uid": uid}).sort("ts", 1)
        
        full_hex_payload = []
        full_decoded_payload = []
        packet_count = 0
        current_len = 0
        
        for p_doc in payload_cursor:
            packet_count += 1
            
            # 截断逻辑：如果超长，只计数不拼接
            if max_payload_len > 0 and current_len >= max_payload_len:
                continue

            if "payload" in p_doc:
                full_hex_payload.append(p_doc["payload"])
            if "payload_decoded" in p_doc:
                decoded_fragment = p_doc["payload_decoded"]
                full_decoded_payload.append(decoded_fragment)
                current_len += len(decoded_fragment)

        # 4. 组装数据
        entry = conn_doc.copy()
        
        # --- 关键：手动清理特殊类型，确保返回的List可以直接被使用 ---
        # 处理 ObjectId
        if "_id" in entry:
            entry["mongo_id"] = str(entry["_id"])
            del entry["_id"]
        
        # 处理 datetime (转为 ISO 格式字符串)
        if "ts_date" in entry and isinstance(entry["ts_date"], datetime):
            entry["ts_date"] = entry["ts_date"].isoformat()

        # 字段赋值
        entry["batch_index"] = processed_count + 1
        entry["packet_count_captured"] = packet_count
        
        # 拼接字符串
        hex_str = "".join(full_hex_payload)
        decoded_str = "".join(full_decoded_payload)
        
        # 最终截断 (确保不超过限制)
        if max_payload_len > 0:
            if len(hex_str) > max_payload_len:
                hex_str = hex_str[:max_payload_len] + "...[TRUNCATED]"
            if len(decoded_str) > max_payload_len:
                decoded_str = decoded_str[:max_payload_len] + "...[TRUNCATED]"
        
        entry["stream_payload_hex"] = hex_str
        entry["stream_payload_decoded"] = decoded_str
        
        export_data.append(entry)
        processed_count += 1

    # 5. 写入文件 (依然保留，作为备份)
    final_output_path = os.path.abspath(output_filename)
    try:
        with open(final_output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, cls=MongoEncoder, ensure_ascii=False, indent=4)
        logging.info(f"文件已保存至: {final_output_path}")
    except Exception as e:
        logging.error(f"文件写入失败: {e}")

    # 6. 直接返回 List
    # 这样你在代码里就可以直接 result[0] 或者 result[:5] 了
    return export_data

if __name__ == "__main__":
    # 手动测试
    result_list = export_flows_tool.invoke({
        "limit": 2, 
        "output_filename": "debug_dataset.json",
        "conn_collection": "conn_test_auto",
        "payload_collection": "payload_test_auto" 
    })
    print(result_list[0])
    print(result_list[1])
    print(f"返回类型: {type(result_list)}") # <class 'list'>
    print(f"获取列表里的第1条数据的uid: {result_list[0].get('uid')}")
    print(f"截取前2条: 长度 {len(result_list[:2])}")