import subprocess
import os
import json
import logging
import sys
from pymongo import MongoClient
from datetime import datetime

# --- 1. 配置路径与环境 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 尝试导入配置
try:
    from utils import config
except ImportError:
    logging.warning("Warning: Could not import utils.config.")
    config = None 

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _hex_to_readable(hex_str):
    """Payload 解码辅助函数"""
    if not hex_str: return ""
    try:
        return bytes.fromhex(hex_str).decode('utf-8', errors='backslashreplace')
    except: return f"<DECODE_ERROR>"

def _run_zeek(pcap_filename):
    """
    运行 Zeek。
    关键逻辑：如果报错信息包含 'truncated dump file'，视为警告而非失败，允许程序继续往下走。
    """
    container = config.DOCKER["zeek_container"]
    
    # Zeek 命令：-C 忽略校验和错误，-r 读取文件
    cmd = [
        "docker", "exec", 
        "-w", "/zeek-logs", 
        container, 
        "zeek", "-C", "-r", f"/pcap/{pcap_filename}", 
        f"/pcap/save-payload.zeek", 
        "policy/tuning/json-logs"
    ]
    
    try:
        # 运行命令
        subprocess.run(cmd, check=True, capture_output=True)
        return True, "Success"
        
    except subprocess.CalledProcessError as e:
        err_output = e.stderr.decode('utf-8', errors='ignore')
        
        # --- 核心容错逻辑 ---
        # 修复后的文件往往还是“截断”的，但 Zeek 此时已经吐出了有效的日志。
        # 所以我们“欺骗”上层函数说这次运行是成功的。
        if "truncated dump file" in err_output:
            return True, f"Truncated Warning (Data preserved): {err_output.splitlines()[0]}"
            
        # 其他错误（如文件找不到、脚本错误）则是真的失败
        return False, err_output

def _ingest(log_name, col_name, db, source_pcap):
    """
    读取日志并入库
    """
    log_path = os.path.join(config.DIRECTORIES["zeek_logs"], log_name)
    
    if not os.path.exists(log_path):
        # 很多时候修复后的包只有 conn 信息，没有 payload，这是正常的
        return 0
    
    collection = db[col_name]
    docs = []
    inserted_count = 0
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    # 标记来源
                    d['source_pcap'] = source_pcap
                    
                    # 格式化时间
                    if 'ts' in d: d['ts_date'] = datetime.fromtimestamp(d['ts'])
                    
                    # 解码 Payload
                    if 'payload' in d: 
                        d['payload_decoded'] = _hex_to_readable(d['payload'])
                        d['payload_size'] = len(d['payload']) // 2
                        
                    docs.append(d)
                    
                    # 批量插入
                    if len(docs) >= 5000:
                        collection.insert_many(docs)
                        inserted_count += len(docs)
                        docs = []
                except json.JSONDecodeError:
                    continue
                    
        # 插入剩余
        if docs:
            collection.insert_many(docs)
            inserted_count += len(docs)
            
        return inserted_count
        
    except Exception as e:
        logging.error(f"入库错误 ({log_name}): {e}")
        return 0

def main():
    json_file = "repaired_files.json"
    conn_col = "3_1_conn"
    payload_col = "3_1_payload"
    
    # 1. 检查修复列表
    if not os.path.exists(json_file):
        logging.error("未找到 repaired_files.json。请确保 Step 2 运行成功。")
        return

    with open(json_file, "r") as f:
        repaired_files = json.load(f)

    if not repaired_files:
        logging.info("修复列表为空，没有文件需要重试。")
        return

    logging.info(f"Step 3: 开始重试 {len(repaired_files)} 个修复后的文件...")

    # 2. 连接数据库
    try:
        client = MongoClient(config.DATABASE["mongo"]["uri"])
        db = client[config.DATABASE["mongo"]["db_name"]]
    except Exception as e:
        logging.error(f"数据库连接失败: {e}")
        return

    success_count = 0
    total_conn = 0
    total_payload = 0
    
    # 3. 循环处理
    for idx, f in enumerate(repaired_files):
        logging.info(f"[{idx+1}/{len(repaired_files)}] 重试分析: {f}")
        
        # A. 运行 Zeek (含容错)
        status, msg = _run_zeek(f)
        
        if status:
            # B. 入库
            c_cnt = _ingest("conn.log", conn_col, db, f)
            p_cnt = _ingest("payload.log", payload_col, db, f)
            
            total_conn += c_cnt
            total_payload += p_cnt
            success_count += 1
            
            # 区分显示是完美成功还是带警告的成功
            if "Truncated" in msg:
                logging.warning(f"  -> 入库成功 (部分截断): Conn={c_cnt}, Payload={p_cnt}")
            else:
                logging.info(f"  -> 入库成功 (完美): Conn={c_cnt}, Payload={p_cnt}")
        else:
            logging.error(f"  -> 重试彻底失败: {msg.strip()[:150]}...")

    # 4. 总结
    logging.info("="*50)
    logging.info(f"Step 3 最终完成报告")
    logging.info(f"处理文件: {len(repaired_files)}")
    logging.info(f"挽回成功: {success_count}")
    logging.info(f"新增 Conn 记录: {total_conn}")
    logging.info(f"新增 Payload 记录: {total_payload}")
    logging.info("="*50)

if __name__ == "__main__":
    main()