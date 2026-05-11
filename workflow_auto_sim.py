import os
import sys
import time
import subprocess
import json
import re
import logging
from pymongo import MongoClient
from datetime import datetime

# --- 动态环境与路径配置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. 确保能导入同级的 utils 和 llm
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 2. 将 zeek-mongo-logs 目录加入系统路径，绕过中划线导包报错问题
zeek_dir = os.path.join(current_dir, "zeek-mongo-lab") 
if os.path.exists(zeek_dir) and zeek_dir not in sys.path:
    sys.path.append(zeek_dir)

# --- 导入自定义模块 ---
from utils import config
from llm import llm  
from step1_analyze import _run_zeek, _ingest, _ensure_script  # 现在可以直接导入了

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Auto-Workflow] - %(message)s')

# --- 数据库配置 ---
MONGO_URI = config.DATABASE["mongo"]["uri"]
DB_NAME = config.GYF_SETTINGS.get("history_db_name", "gyf_history")
INTEL_COLLECTION = "xlab_threat_intel"  # 爬虫存入的集合
CONN_COLLECTION = "sim_c2_conn"         # 生成流量的存放集合
PAYLOAD_COLLECTION = "sim_c2_payload"

PCAP_DIR = os.path.join(current_dir, "pcap")
os.makedirs(PCAP_DIR, exist_ok=True)

# 抓包模板，用于指导 LLM 生成代码结构
SIMULATION_TEMPLATE = """
请参考以下 3合1 (Server, Client, Sniffer) 的 Python 抓包脚本结构：
1. start_mock_server(): 监听特定端口，模拟 C2 服务端行为。
2. start_mock_client(): 连接服务端，模拟被控端发送上线包和心跳。
3. capture_traffic(): 使用 scapy 的 sniff 监听 lo 网卡，捕获交互数据并保存为 pcap。
"""

def fetch_unprocessed_intel(db):
    """从数据库获取尚未生成过流量的情报文档"""
    # 假设我们用一个标记 'simulated' 来区分
    query = {"simulated": {"$ne": True}}
    return list(db[INTEL_COLLECTION].find(query))

def generate_simulation_script(intel_doc):
    """交由大模型生成模拟脚本"""
    title = intel_doc.get("title", "Unknown")
    content = intel_doc.get("content", "")
    pcap_filename = f"sim_{intel_doc['_id']}.pcap"
    
    prompt = f"""你是一个高级网络安全专家。请根据以下威胁情报的内容，编写一个 Python 3合1 流量模拟脚本 (C2服务端 + 被控端 + 抓包器)。
    
【威胁情报内容】：
{content[:200000]} # 截取前200000字符避免超长

【要求】：
1. 提取情报中的通信协议特征（如端口、特定的 Magic Bytes、加密方式等）。
2. {SIMULATION_TEMPLATE}
3. PCAP 文件名必须硬编码为 "{pcap_filename}"，并保存在当前目录下。
4. 确保 server 和 client 在子线程中运行，主线程阻塞执行 sniff 抓包。
5. 只输出 Python 代码，不要输出其他 Markdown 解释。
"""
    
    logging.info(f"正在请求 LLM 生成脚本，针对情报: {title}")
    response = llm.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    
    # 提取纯代码块
    match = re.search(r'```python\s*(.*?)\s*```', response_text, re.DOTALL)
    if match:
        return match.group(1), pcap_filename
    return response_text, pcap_filename

def run_simulation(script_code, pcap_filename):
    """执行脚本并生成 PCAP"""
    script_path = os.path.join(PCAP_DIR, f"temp_sim_{int(time.time())}.py")
    
    # 写入临时脚本
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script_code)
        
    logging.info(f"开始执行模拟脚本: {script_path}")
    try:
        # 在 pcap 目录下执行，确保生成的 pcap 文件落入该目录
        subprocess.run([sys.executable, script_path], cwd=PCAP_DIR, timeout=30, check=True)
        expected_pcap_path = os.path.join(PCAP_DIR, pcap_filename)
        
        if os.path.exists(expected_pcap_path):
            logging.info(f"成功生成 PCAP: {expected_pcap_path}")
            return expected_pcap_path
        else:
            logging.error("脚本执行完毕，但未找到预期的 PCAP 文件。")
            return None
    except subprocess.TimeoutExpired:
        logging.error("执行脚本超时！")
        return None
    except Exception as e:
        logging.error(f"执行脚本异常: {e}")
        return None

def analyze_and_ingest(pcap_filename, db, intel_doc):
    """调用 Zeek 分析并入库，同时打上知识库外键"""
    logging.info(f"开始 Zeek 解析: {pcap_filename}")
    
    zeek_script_path = os.path.join(PCAP_DIR, "save-payload.zeek")
    _ensure_script(zeek_script_path)
    
    success, msg = _run_zeek(pcap_filename)
    if not success:
        logging.error(f"Zeek 解析失败: {msg}")
        return False
        
    # 1. 正常入库（此时有了 source_pcap 字段）
    conn_count = _ingest("conn.log", CONN_COLLECTION, db, pcap_filename)
    payload_count = _ingest("payload.log", PAYLOAD_COLLECTION, db, pcap_filename)
    
    # 2. 🌟 核心优化：批量更新这批刚刚入库的流量，显式加上知识库的关联字段
    if conn_count > 0 or payload_count > 0:
        update_data = {
            "$set": {
                "intel_id": intel_doc["_id"],            # 存入原始情报的 _id
                "intel_title": intel_doc.get("title")    # 顺便冗余一个标题，方便人类查看
            }
        }
        # 更新 conn 表
        db[CONN_COLLECTION].update_many({"source_pcap": pcap_filename}, update_data)
        # 更新 payload 表
        db[PAYLOAD_COLLECTION].update_many({"source_pcap": pcap_filename}, update_data)
    
    logging.info(f"入库并绑定知识库完成: conn={conn_count}条, payload={payload_count}条")
    return True

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    intel_docs = fetch_unprocessed_intel(db)
    logging.info(f"发现 {len(intel_docs)} 篇待处理情报。")
    
    for doc in intel_docs:
        doc_id = doc["_id"]
        title = doc.get("title", "Unknown")
        
        # 1. 生成代码
        code, pcap_filename = generate_simulation_script(doc)
        if "scapy" not in code or "socket" not in code:
            logging.warning(f"情报 {title} 生成的代码可能无效，跳过。")
            continue
            
        # 2. 执行并抓包
        pcap_path = run_simulation(code, pcap_filename)
        if not pcap_path:
            continue
            
        # 3. 流量解析与入库
        if analyze_and_ingest(pcap_filename, db, doc):
            # 4. 更新情报状态，标记为已模拟
            db[INTEL_COLLECTION].update_one({"_id": doc_id}, {"$set": {"simulated": True}})
            logging.info(f"✅ 情报 [{title}] 闭环处理完成！\n" + "="*50)
            
        time.sleep(2)

if __name__ == "__main__":
    main()