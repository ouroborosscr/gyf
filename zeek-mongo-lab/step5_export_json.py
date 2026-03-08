import json
import os
import sys
import logging
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime

# --- 环境设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from utils import config
except ImportError:
    logging.error("无法导入 utils.config，请确保路径正确")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置 ---
CONN_COL = "conn_gyf_demo"    # 你的 conn 集合名
PAYLOAD_COL = "payload_gyf_demo" # 你的 payload 集合名
OUTPUT_FILE = "label_studio_tasks.json"

# JSON 编码器，处理 ObjectId 和 datetime
class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def format_conversation(conn, payloads):
    """
    生成一段易于人工阅读的文本，模拟对话流。
    Label Studio 将直接展示这段文本供人工判断。
    """
    lines = []
    
    # 获取 IP 信息 (兼容 Step 3.1 的下划线命名 或 原始的点命名)
    src_ip = conn.get("id_orig_h", conn.get("id.orig_h", "Unknown"))
    dst_ip = conn.get("id_resp_h", conn.get("id.resp_h", "Unknown"))
    src_port = conn.get("id_orig_p", conn.get("id.orig_p", 0))
    dst_port = conn.get("id_resp_p", conn.get("id.resp_p", 0))

    # 头部信息
    header = f"Session UID: {conn.get('uid')}\n"
    header += f"Protocol: {conn.get('proto')} | Service: {conn.get('service', 'none')}\n"
    header += f"{src_ip}:{src_port}  -->  {dst_ip}:{dst_port}\n"
    header += "=" * 50
    lines.append(header)

    # 对 Payload 按时间排序
    sorted_payloads = sorted(payloads, key=lambda x: x.get('ts', 0))

    for p in sorted_payloads:
        # 判断方向
        is_orig = p.get('is_orig', True)
        direction = "-->" if is_orig else "<--"
        sender = src_ip if is_orig else dst_ip
        
        # 获取内容
        content = p.get('payload_decoded', '')
        if not content:
            content = f"[HEX ONLY]: {p.get('payload', '')[:50]}..."
        
        # 组装行
        timestamp = datetime.fromtimestamp(p.get('ts', 0)).strftime('%H:%M:%S.%f')[:-3]
        lines.append(f"\n[{timestamp}] {sender} {direction} ({p.get('len', 0)} bytes):")
        lines.append(f"{content}")
        lines.append("-" * 20)

    return "\n".join(lines)

def main():
    try:
        client = MongoClient(config.DATABASE["mongo"]["uri"])
        db = client[config.DATABASE["mongo"]["db_name"]]
        conn_collection = db[CONN_COL]

        logging.info("开始聚合数据...")

        # 使用 MongoDB 聚合管道进行联表查询 (Lookup)
        # 逻辑：查找 conn -> 关联 payload -> 生成结果
        pipeline = [
            # 1. 筛选条件：这里可以只筛选有 payload 的流，或者筛选特定协议
            # 如果不需要筛选，注释掉下面这行
            # { "$match": { "service": "http" } }, 

            # 2. 联表查询
            {
                "$lookup": {
                    "from": PAYLOAD_COL,
                    "localField": "uid",
                    "foreignField": "uid",
                    "as": "related_payloads"
                }
            },

            # 3. 可以在这里做一些投影，减少数据量，但为了完整性先保留
        ]

        cursor = conn_collection.aggregate(pipeline)
        
        tasks = []
        count = 0

        for doc in cursor:
            # 提取 payload
            payloads = doc.get("related_payloads", [])
            
            # 如果只想打标有内容的流，可以取消下面的注释
            # if not payloads: continue

            # 生成 Label Studio 专用结构
            # Label Studio 要求数据放在 "data" 字段下
            task = {
                "data": {
                    "uid": doc.get("uid"),
                    "info": f"{doc.get('proto')} {doc.get('service', '')}",
                    # 原始 conn 数据
                    "conn_details": doc, 
                    # 原始 payload 列表 (保留供通过脚本分析)
                    "payloads_raw": payloads,
                    # ***核心***：生成可视化文本
                    "conversation_text": format_conversation(doc, payloads)
                }
            }
            tasks.append(task)
            count += 1
            
            if count % 1000 == 0:
                logging.info(f"已处理 {count} 条会话...")

        # 写入文件
        logging.info(f"正在写入 {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            # ensure_ascii=False 保证中文和特殊字符正常显示
            json.dump(tasks, f, cls=MongoEncoder, ensure_ascii=False, indent=2)

        logging.info(f"完成！共生成 {len(tasks)} 个打标任务。")
        logging.info("请将生成的 json 文件导入 Label Studio。")

    except Exception as e:
        logging.exception(f"执行出错: {e}")

if __name__ == "__main__":
    main()