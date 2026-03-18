import subprocess
import os
import json
import logging
import sys
from pymongo import MongoClient
from datetime import datetime
from langchain.tools import tool

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

def _hex_to_readable(hex_str: str) -> str:
    """
    [来自 hex2utf8.py 的逻辑]
    将 hex 字符串转换为“人类可读”的形式。
    """
    if not hex_str:
        return ""
    
    try:
        # 1. 16进制 -> 二进制
        byte_data = bytes.fromhex(hex_str)
        # 2. 二进制 -> 字符串 (使用 backslashreplace 处理无法解码的字符)
        readable_text = byte_data.decode('utf-8', errors='backslashreplace')
        return readable_text
    except Exception as e:
        # 保持静默错误处理，避免打断整个入库流程，但记录日志
        return f"<DECODE_ERROR: {hex_str}>"

def _run_zeek_in_container(pcap_filename: str):
    """
    内部辅助函数：调用 Docker 执行 Zeek
    """
    container_name = config.DOCKER["zeek_container"]
    zeek_script = config.ZEEK["default_script"]
    
    # 路径映射
    container_pcap_path = f"/pcap/{pcap_filename}"
    container_script_path = f"/pcap/{zeek_script}"
    container_log_dir = "/zeek-logs"

    logging.info(f"正在容器 {container_name} 中处理 {pcap_filename} ...")

    cmd = [
        "docker", "exec",
        "-w", container_log_dir,  
        container_name,
        "zeek", "-C", "-r", container_pcap_path,
        container_script_path, 
        "policy/tuning/json-logs"
    ]
    
    try:
        logging.info(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        return True, "Zeek processing completed successfully."
    except subprocess.CalledProcessError as e:
        error_msg = f"Zeek 运行失败: {e.stderr.decode('utf-8') if e.stderr else str(e)}"
        logging.error(error_msg)
        return False, error_msg

def _ingest_log_to_mongo(log_filename, collection_name, db):
    """
    内部辅助函数：读取日志，(可选)进行解码清洗，存入 Mongo
    """
    log_dir = config.DIRECTORIES["zeek_logs"]
    file_path = os.path.join(log_dir, log_filename)

    if not os.path.exists(file_path):
        return 0, f"Log file not found: {file_path}"

    collection = db[collection_name]
    json_docs = []
    inserted_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    
                    # --- 1. 标准时间戳处理 ---
                    if 'ts' in data:
                        data['ts_date'] = datetime.fromtimestamp(data['ts'])
                    
                    # --- 2. [新增] Payload 解码处理 ---
                    # 检查是否存在 payload 字段 (通常在 payload.log 中)
                    if 'payload' in data:
                        # 执行解码
                        data['payload_decoded'] = _hex_to_readable(data['payload'])
                        # 增加长度统计
                        data['payload_size'] = len(data['payload']) // 2
                    
                    json_docs.append(data)
                    
                    # 批量插入
                    if len(json_docs) >= 5000:
                        collection.insert_many(json_docs)
                        inserted_count += len(json_docs)
                        json_docs = []
                except json.JSONDecodeError:
                    continue
        
        # 插入剩余数据
        if json_docs:
            collection.insert_many(json_docs)
            inserted_count += len(json_docs)
            
        return inserted_count, f"Successfully ingested {inserted_count} records into {collection_name}."
        
    except Exception as e:
        return 0, f"Error ingesting {collection_name}: {str(e)}"

# --- Tool 定义 ---

@tool
def analyze_pcap_tool(
    pcap_filename: str = None, 
    conn_collection: str = "connections", 
    payload_collection: str = "payloads"
) -> str:
    """
    使用 Zeek 分析 PCAP 文件，自动解码 Payload 十六进制数据，并将结果存入 MongoDB。
    
    该工具会自动处理以下任务：
    1. 运行 Zeek 分析指定的流量包。
    2. 解析 conn.log 并存入指定集合。
    3. 解析 payload.log，将十六进制 payload 转换为可读文本 (payload_decoded)，存入指定集合。
    
    Args:
        pcap_filename (str, optional): 需要分析的 PCAP 文件名。默认使用配置中的文件。
        conn_collection (str, optional): 连接日志集合名。默认为 "connections"。
        payload_collection (str, optional): 负载日志集合名。默认为 "payloads"。
        
    Returns:
        str: 执行结果摘要，包含入库记录数。
    """
    logging.info("正在使用 zeek 工具")
    
    # 1. 确定文件名
    target_pcap = pcap_filename if pcap_filename else config.ZEEK["default_pcap"]
    result_summary = []

    # 2. 连接数据库
    try:
        mongo_cfg = config.DATABASE["mongo"]
        client = MongoClient(mongo_cfg["uri"])
        db = client[mongo_cfg["db_name"]]
    except Exception as e:
        return f"Database Connection Failed: {str(e)}"

    # 3. 运行 Zeek
    success, msg = _run_zeek_in_container(target_pcap)
    result_summary.append(f"Analysis Step: {msg}")
    
    if not success:
        return "\n".join(result_summary)

    # 4. 入库 conn.log
    count_conn, msg_conn = _ingest_log_to_mongo("conn.log", conn_collection, db)
    result_summary.append(f"DB Ingestion ({conn_collection}): {msg_conn}")

    # 5. 入库 payload.log (会自动触发 _hex_to_readable)
    count_payload, msg_payload = _ingest_log_to_mongo("payload.log", payload_collection, db)
    result_summary.append(f"DB Ingestion ({payload_collection}): {msg_payload}")

    total_records = count_conn + count_payload
    final_output = "\n".join(result_summary)
    
    return f"Job Completed. Processed '{target_pcap}'. Total records inserted: {total_records}.\nDetails:\n{final_output}"

if __name__ == "__main__":
    # 手动测试
    print(analyze_pcap_tool.invoke({
        "pcap_filename": "clean.pcap",
        "conn_collection": "conn_Thursday",
        "payload_collection": "payload_Thursday"
    }))