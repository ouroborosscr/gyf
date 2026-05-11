import copy
import re
import numpy as np
from pymongo import MongoClient
import logging

# 导入项目中现有的模块
from utils import config
from llm import embeddings
from rag import _build_batch_profile, _extract_payload_summary, _cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EVAL] - %(message)s")

class InMemoryMasker:
    """内存动态脱敏器"""
    def __init__(self):
        self.ip_map = {}
        self.ip_counter = 1

    def get_fake_ip(self, real_ip):
        if real_ip not in self.ip_map:
            self.ip_map[real_ip] = f"[脱敏IP_{self.ip_counter}]"
            self.ip_counter += 1
        return self.ip_map[real_ip]

    def mask_flows(self, flows):
        masked_flows = []
        for original_f in flows:
            f = copy.deepcopy(original_f) 
            if "id.orig_h" in f:
                f["id.orig_h"] = self.get_fake_ip(f["id.orig_h"])
            if "id.resp_h" in f:
                f["id.resp_h"] = self.get_fake_ip(f["id.resp_h"])

            payload = f.get("stream_payload_decoded", "")
            if payload:
                ip_pattern = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
                found_ips = set(re.findall(ip_pattern, payload))
                for ip in found_ips:
                    payload = payload.replace(ip, self.get_fake_ip(ip))
                payload = re.sub(r':(\d{2,5})\b', r':[脱敏端口]', payload)
                f["stream_payload_decoded"] = payload

            masked_flows.append(f)
        return masked_flows

def load_merged_flows(db, prefix):
    """从 MongoDB 中加载并合并 conn 和 payload 集合，带智能容错与 Debug 打印"""
    conn_col = db[f"{prefix}_conn"]
    payload_col = db[f"{prefix}_payload"]
    
    conns = list(conn_col.find({}))
    if not conns:
        logging.warning(f"数据集 {prefix}_conn 为空！")
        return []
        
    uids = [c.get("uid") for c in conns if c.get("uid")]
    payloads = list(payload_col.find({"uid": {"$in": uids}}))
    payload_dict = {p["uid"]: p for p in payloads if "uid" in p}
    
    merged_flows = []
    payload_match_count = 0
    total_payload_length = 0

    for c in conns:
        uid = c.get("uid")
        if uid in payload_dict:
            p_doc = payload_dict[uid]
            # 智能容错：尝试多种常见的 Payload 字段名
            payload_text = p_doc.get("stream_payload_decoded") or p_doc.get("payload") or p_doc.get("data") or ""
            
            # 确保是字符串
            if isinstance(payload_text, bytes):
                payload_text = payload_text.decode('utf-8', errors='ignore')
            elif not isinstance(payload_text, str):
                payload_text = str(payload_text)

            c["stream_payload_decoded"] = payload_text

            if payload_text.strip():
                payload_match_count += 1
                total_payload_length += len(payload_text)

        merged_flows.append(c)
        
    logging.info(f" -> [{prefix}] 共 {len(conns)} 条流，其中 {payload_match_count} 条成功关联到非空 Payload (总字符数: {total_payload_length})")
    
    if payload_match_count == 0:
        logging.warning(f" ⚠️ 警告: [{prefix}] 的 Payload 全为空！请检查 {prefix}_payload 集合中的字段名。")

    return merged_flows

def get_flow_feature_vector(flows):
    """提取融合向量，并返回行为统计维度的真实长度"""
    if not flows:
        return np.zeros(2606, dtype=np.float32), 46

    # 1. 行为统计特征
    stats_vec = _build_batch_profile(flows)
    actual_stats_dim = len(stats_vec)  # 动态获取实际维度（修复 rag.py 维度不对齐的问题）

    # 2. 文本 Payload 特征
    payload_text = _extract_payload_summary(flows)
    
    if payload_text:
        payload_vec_raw = embeddings.embed_query(payload_text)
        payload_vec = np.array(payload_vec_raw, dtype=np.float32)
        payload_vec = payload_vec / (np.linalg.norm(payload_vec) or 1.0)
    else:
        payload_vec = np.zeros(2560, dtype=np.float32) 
        
    WEIGHT_STATS = 0.6
    WEIGHT_PAYLOAD = 0.4
    
    combined_vec = np.concatenate([
        np.sqrt(WEIGHT_STATS) * stats_vec,
        np.sqrt(WEIGHT_PAYLOAD) * payload_vec
    ])
    
    norm = np.linalg.norm(combined_vec)
    final_vec = combined_vec / norm if norm > 0 else combined_vec

    return final_vec, actual_stats_dim

def main():
    uri = config.DATABASE["mongo"]["uri"]
    client = MongoClient(uri)
    db = client[config.DATABASE["mongo"]["db_name"]]

    target_prefix = "2_true"
    candidate_prefixes = ["1", "2", "rimasuta_c2", "smargaft_c2"]
    masker = InMemoryMasker()

    logging.info(f"正在加载待验证流量 (Target): {target_prefix}")
    target_flows = load_merged_flows(db, target_prefix)
    masked_target_flows = masker.mask_flows(target_flows)
    target_vec, stats_dim = get_flow_feature_vector(masked_target_flows)

    if np.all(target_vec == 0):
        logging.error("目标流量向量全为 0，请检查 1_true_conn 集合是否有数据！")
        return

    results = []
    logging.info("开始计算候选集相似度...")
    
    for cand in candidate_prefixes:
        cand_flows = load_merged_flows(db, cand)
        masked_cand_flows = masker.mask_flows(cand_flows)
        cand_vec, _ = get_flow_feature_vector(masked_cand_flows)
        
        # 使用动态修正后的 stats_dim 进行粗排切片比较
        coarse_sim = _cosine_similarity(target_vec[:stats_dim], cand_vec[:stats_dim])
        fine_sim = _cosine_similarity(target_vec, cand_vec)
        
        results.append({
            "dataset": cand,
            "coarse_sim": coarse_sim,
            "fine_sim": fine_sim
        })

    results.sort(key=lambda x: x["fine_sim"], reverse=True)

    print("\n" + "="*80)
    print(f"🎯 流匹配召回验证结果 (目标: {target_prefix}) [内存脱敏 + 维度修复]")
    print("="*80)
    print(f"{'排名':<4} | {'候选数据集':<15} | {'精排相似度(完整)':<18} | {'粗排相似度(仅行为)'}")
    print("-" * 80)

    target_rank = -1
    for idx, res in enumerate(results, 1):
        if res['dataset'] == "2":
            target_rank = idx
            marker = " <--- [我们要找的目标]"
        else:
            marker = ""
            
        print(f"#{idx:<3} | {res['dataset']:<15} | {res['fine_sim']:<18.4f} | {res['coarse_sim']:.4f}{marker}")

    print("="*80)
    if target_rank != -1:
        print(f"💡 结论: 正确的关联数据集 '2' 召回排在第 【 {target_rank} 】 位。")
    else:
        print("💡 结论: 未在候选集中找到数据集 '2'。")

if __name__ == "__main__":
    main()