import subprocess
import os
import json
import logging
import sys
from pymongo import MongoClient
from datetime import datetime

# --- 配置与环境 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from utils import config
except ImportError:
    config = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 辅助函数 ---
def _hex_to_readable(hex_str):
    if not hex_str: return ""
    try:
        return bytes.fromhex(hex_str).decode('utf-8', errors='backslashreplace')
    except: return f"<DECODE_ERROR>"

def _ensure_script(local_path):
    if not os.path.exists(local_path): return False
    cmd = ["docker", "cp", local_path, f"{config.DOCKER['zeek_container']}:/pcap/save-payload.zeek"]
    return subprocess.run(cmd, capture_output=True).returncode == 0

def _run_zeek(pcap_filename):
    container = config.DOCKER["zeek_container"]
    cmd = ["docker", "exec", "-w", "/zeek-logs", container, "zeek", "-C", "-r", f"/pcap/{pcap_filename}", f"/pcap/save-payload.zeek", "policy/tuning/json-logs"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True, "Success"
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode('utf-8')
        if "truncated dump file" in err: return True, "Truncated (Warning)" # 容错
        return False, err

def _ingest(log_name, col_name, db, source):
    path = os.path.join(config.DIRECTORIES["zeek_logs"], log_name)
    if not os.path.exists(path): return 0
    
    col = db[col_name]
    docs = []
    count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
                d['source_pcap'] = source
                if 'ts' in d: d['ts_date'] = datetime.fromtimestamp(d['ts'])
                if 'payload' in d: d['payload_decoded'] = _hex_to_readable(d['payload'])
                docs.append(d)
                if len(docs) >= 5000:
                    col.insert_many(docs)
                    count += len(docs)
                    docs = []
            except: continue
    if docs:
        col.insert_many(docs)
        count += len(docs)
    return count

# --- 主逻辑 ---
def main():
    pcap_dir = "./pcap"
    conn_col = "3_1_conn"
    payload_col = "3_1_payload"
    
    # 1. 准备文件列表
    files = [f for f in os.listdir(pcap_dir) if f != "save-payload.zeek" and not f.startswith(".") and os.path.isfile(os.path.join(pcap_dir, f))]
    logging.info(f"Step 1: 发现 {len(files)} 个文件。")

    # 2. 确保脚本在容器内
    _ensure_script(os.path.join(pcap_dir, "save-payload.zeek"))

    # 3. 连接数据库
    client = MongoClient(config.DATABASE["mongo"]["uri"])
    db = client[config.DATABASE["mongo"]["db_name"]]

    failed_files = []
    
    for idx, f in enumerate(files):
        logging.info(f"[{idx+1}/{len(files)}] Processing {f}...")
        success, msg = _run_zeek(f)
        
        if not success:
            logging.error(f"Failed: {f} -> {msg.strip()[:100]}")
            failed_files.append(f)
            continue
            
        _ingest("conn.log", conn_col, db, f)
        _ingest("payload.log", payload_col, db, f)

    # 4. 导出失败列表供下一步使用
    with open("failed_files.json", "w") as f:
        json.dump(failed_files, f)
    
    logging.info(f"Step 1 完成. 成功: {len(files)-len(failed_files)}, 失败: {len(failed_files)} (已写入 failed_files.json)")

if __name__ == "__main__":
    main()