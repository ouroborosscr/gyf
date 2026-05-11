import numpy as np
from pymongo import MongoClient
import logging

# 导入项目中现有的模块
from utils import config
from llm import embeddings
from rag import _build_batch_profile, _extract_payload_summary, _cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EVAL] - %(message)s")

def load_merged_flows(db, prefix):
    """
    从 MongoDB 中加载并合并 conn 和 payload 集合
    模拟 rag.py 中完整的 flow 对象
    """
    conn_col = db[f"{prefix}_conn"]
    payload_col = db[f"{prefix}_payload"]
    
    # 1. 获取所有连接记录
    conns = list(conn_col.find({}))
    if not conns:
        logging.warning(f"数据集 {prefix}_conn 为空！")
        return []
        
    # 2. 获取对应的 payload 记录
    uids = [c.get("uid") for c in conns if c.get("uid")]
    payloads = list(payload_col.find({"uid": {"$in": uids}}))
    payload_dict = {p["uid"]: p for p in payloads if "uid" in p}
    
    # 3. 合并到流字典中
    merged_flows = []
    for c in conns:
        uid = c.get("uid")
        if uid in payload_dict:
            # rag.py 的 _extract_payload_summary 依赖 stream_payload_decoded 字段
            c["stream_payload_decoded"] = payload_dict[uid].get("stream_payload_decoded", "")
        merged_flows.append(c)
        
    return merged_flows

def get_flow_feature_vector(flows):
    """
    完全复用 rag.py 的节点2 (save_think) 的特征融合逻辑
    提取 2606 维融合向量
    """
    if not flows:
        return np.zeros(2606, dtype=np.float32)

    # 1. 提取 46 维数值特征 (L2已归一化)
    stats_vec = _build_batch_profile(flows)
    
    # 2. 提取 payload 文本并做 embedding (2560维)
    payload_text = _extract_payload_summary(flows)
    if payload_text:
        payload_vec_raw = embeddings.embed_query(payload_text)
        payload_vec = np.array(payload_vec_raw, dtype=np.float32)
        payload_vec = payload_vec / np.linalg.norm(payload_vec)
    else:
        payload_vec = np.zeros(2560, dtype=np.float32) 
        
    # 3. 权重拼接 (行为 0.6 : 载荷 0.4)
    WEIGHT_STATS = 0.6
    WEIGHT_PAYLOAD = 0.4
    
    combined_vec = np.concatenate([
        np.sqrt(WEIGHT_STATS) * stats_vec,
        np.sqrt(WEIGHT_PAYLOAD) * payload_vec
    ])
    
    # 二次归一化兜底
    norm = np.linalg.norm(combined_vec)
    return combined_vec / norm if norm > 0 else combined_vec

def main():
    # 1. 连接数据库 (使用 zeek_db)
    uri = config.DATABASE["mongo"]["uri"]
    client = MongoClient(uri)
    db = client[config.DATABASE["mongo"]["db_name"]]

    # 2. 定义测试集
    target_prefix = "2_true"
    candidate_prefixes = ["1", "2", "rimasuta_c2", "smargaft_c2"]

    # 3. 提取目标流量 (Ground Truth) 的特征向量
    logging.info(f"正在加载待验证流量 (Target): {target_prefix}")
    target_flows = load_merged_flows(db, target_prefix)
    target_vec = get_flow_feature_vector(target_flows)

    if np.all(target_vec == 0):
        logging.error("目标流量向量全为 0，请检查 2_true_conn 集合是否有数据！")
        return

    results = []

    # 4. 计算与各个候选集的相似度
    logging.info("开始计算候选集相似度...")
    for cand in candidate_prefixes:
        logging.info(f" -> 处理候选集: {cand}")
        cand_flows = load_merged_flows(db, cand)
        cand_vec = get_flow_feature_vector(cand_flows)
        
        # rag.py 中的粗排逻辑 (仅比较前46维网络行为)
        coarse_sim = _cosine_similarity(target_vec[:46], cand_vec[:46])
        # rag.py 中的精排逻辑 (比较完整2606维)
        fine_sim = _cosine_similarity(target_vec, cand_vec)
        
        results.append({
            "dataset": cand,
            "coarse_sim": coarse_sim,
            "fine_sim": fine_sim
        })

    # 5. 按精排相似度降序排序
    results.sort(key=lambda x: x["fine_sim"], reverse=True)

    # 6. 打印报告
    print("\n" + "="*70)
    print(f"🎯 流匹配召回验证结果 (目标: {target_prefix})")
    print("="*70)
    print(f"{'排名':<4} | {'候选数据集':<15} | {'精排相似度(完整)':<18} | {'粗排相似度(仅行为)'}")
    print("-" * 70)

    target_rank = -1
    for idx, res in enumerate(results, 1):
        if res['dataset'] == "2":
            target_rank = idx
            marker = " <--- [我们要找的目标]"
        else:
            marker = ""
            
        print(f"#{idx:<3} | {res['dataset']:<15} | {res['fine_sim']:<18.4f} | {res['coarse_sim']:.4f}{marker}")

    print("="*70)
    if target_rank != -1:
        print(f"💡 结论: 正确的关联数据集 '2' 召回排在第 【 {target_rank} 】 位。")
    else:
        print("💡 结论: 未在候选集中找到数据集 '2'。")

if __name__ == "__main__":
    main()