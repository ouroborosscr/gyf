import os
import sys
import json
import logging
from pymongo import MongoClient

# --- 动态添加主目录到环境变量，解决跨目录导包问题 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入主目录的依赖
from intelligence.exp2_retrieval import TARGET_TITLE
from llm import llm
from utils import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EXP1] - %(message)s")

# --- 配置区 ---
MONGO_URI = config.DATABASE["mongo"]["uri"]
DB_NAME = config.DATABASE["mongo"]["db_name"]
HISTORY_DB_NAME = config.GYF_SETTINGS.get("history_db_name", "gyf_history")

# 你抓取的 PCAP 流存入的集合名
PCAP_COL_NAME = "rimasuta_c2_conn" 
# PCAP_COL_NAME = "smargaft_c2_conn"
TARGET_TITLE = "Rimasuta新变种出现，改用ChaCha20加密"
# TARGET_TITLE = "Smargaft Harnesses EtherHiding for Stealthy C2 Hosting"

def get_pcap_flows(db, limit=50):
    """提取 PCAP 流量、转换为标准安全术语，并进行一致性 IP 脱敏"""
    col = db[PCAP_COL_NAME]
    raw_flows = list(col.find({}, {"_id": 0}).limit(limit))
    
    clean_flows = []
    
    # 🌟 新增：IP 脱敏映射表
    ip_map = {}
    
    def get_masked_ip(ip_str):
        """将真实 IP 转换为 Masked_IP_X，且同一个真实 IP 始终对应同一个假名"""
        if not ip_str: 
            return "UNKNOWN_IP"
        if ip_str not in ip_map:
            # 按照发现的先后顺序编号
            ip_map[ip_str] = f"Masked_IP_{len(ip_map) + 1}"
        return ip_map[ip_str]

    for f in raw_flows:
        # 提取真实 IP
        real_src = f.get("id_orig_h") or f.get("id.orig_h")
        real_dst = f.get("id_resp_h") or f.get("id.resp_h")
        
        clean_flows.append({
            "uid": f.get("uid"),
            "start_time": f.get("ts"),
            
            # 🌟 使用映射函数输出脱敏后的 IP
            "src_ip": get_masked_ip(real_src),
            "dst_ip": get_masked_ip(real_dst),
            
            "src_port": f.get("id_orig_p") or f.get("id.orig_p"),
            "dst_port": f.get("id_resp_p") or f.get("id.resp_p"),
            
            "proto": f.get("proto"),
            "duration": f.get("duration"),
            "orig_bytes": f.get("orig_bytes"),
            "resp_bytes": f.get("resp_bytes"),
            
            "payload": f.get("stream_payload_decoded", "")[:300] 
        })
        
    logging.info(f"🔒 流量 IP 脱敏完成。共映射了 {len(ip_map)} 个独立 IP。")
    return clean_flows

def get_threat_intel(db):
    """从本地知识库中获取对应的威胁情报文章"""
    col = db["xlab_threat_intel"]
    doc = col.find_one({"title": TARGET_TITLE})
    if not doc:
        logging.error(f"未找到目标文章，请确认 spider_xlab.py 已成功抓取且标题一致")
        sys.exit(1)
    return doc["content"]

def main():
    client = MongoClient(MONGO_URI)
    zeek_db = client[DB_NAME]
    history_db = client[HISTORY_DB_NAME]

    logging.info("1. 正在提取 PCAP 流量数据并进行脱敏...")
    flows = get_pcap_flows(zeek_db)
    if not flows:
        logging.error(f"集合 {PCAP_COL_NAME} 中没有数据！请先运行 step1-3 处理 PCAP。")
        return
    flows_str = json.dumps(flows, ensure_ascii=False, indent=2)

    logging.info("2. 正在提取 XLab 威胁情报...")
    intel_content = get_threat_intel(history_db)

    # ---------------- 阶段 1：Baseline (裸判) ----------------
    logging.info("3. 开始 Baseline 研判 (无知识库)...")
    baseline_prompt = f"""你是一名网络安全分析专家。以下是我截获的一批网络流量记录（JSON格式）。
请分析这些流量，判断其是否属于 C2 攻击。

【流量数据】：
{flows_str}

请给出明确的研判结论和依据。"""

    baseline_response = llm.invoke(baseline_prompt)
    baseline_result = baseline_response.content if hasattr(baseline_response, "content") else str(baseline_response)
    
    print("\n" + "="*40 + " Baseline (无知识库) 结果 " + "="*40)
    print(baseline_result)
    print("="*106 + "\n")

    # 将 Baseline 的结果保存下来，供实验 2 使用
    with open(PCAP_COL_NAME+"_baseline_result.txt", "w", encoding="utf-8") as f:
        f.write(baseline_result)

    # ---------------- 阶段 2：RAG 增强研判 ----------------
    logging.info("4. 开始 RAG 增强研判 (注入 XLab 情报)...")
    rag_prompt = f"""你是一名网络安全分析专家。以下是我截获的一批网络流量记录（JSON格式），以及从安全知识库中检索到的一份【威胁情报参考】。
请结合该威胁情报，分析这些流量，判断其是否属于 C2 攻击。

【威胁情报参考】：
{intel_content}

【流量数据】：
{flows_str}

请结合情报中的特征（如端口、加密算法特征、通信模式等），给出明确的研判结论和依据。"""

    rag_response = llm.invoke(rag_prompt)
    rag_result = rag_response.content if hasattr(rag_response, "content") else str(rag_response)

    print("\n" + "="*42 + " RAG (注入知识库) 结果 " + "="*42)
    print(rag_result)
    print("="*106 + "\n")

if __name__ == "__main__":
    main()