import os
import sys
import re
import numpy as np
import logging
from pymongo import MongoClient
import jieba
from rank_bm25 import BM25Okapi

# --- 动态添加主目录到环境变量，解决跨目录导包问题 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 导入主目录的依赖
from llm import embeddings
from utils import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [EXP2] - %(message)s")

MONGO_URI = config.DATABASE["mongo"]["uri"]
HISTORY_DB_NAME = config.GYF_SETTINGS.get("history_db_name", "gyf_history")
PCAP_COL_NAME = "rimasuta_c2_conn" 
# PCAP_COL_NAME = "smargaft_c2_conn"
TARGET_TITLE = "Rimasuta新变种出现，改用ChaCha20加密"
# TARGET_TITLE = "Smargaft Harnesses EtherHiding for Stealthy C2 Hosting"

# 复用 rag.py 的分词逻辑
def _tokenize(text: str) -> list:
    if not text or not text.strip(): return ["__empty__"]
    raw_tokens = jieba.lcut(text)
    result = [t.strip().lower() for t in raw_tokens if t.strip() and not re.match(r'^[^\w\s\u4e00-\u9fff]+$', t.strip())]
    return result if result else ["__empty__"]

# 复用 rag.py 的余弦相似度
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.shape != b.shape: return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def main():
    # 1. 读取 Baseline 的大模型分析结果作为 Query
    if not os.path.exists(PCAP_COL_NAME+"_baseline_result.txt"):
        logging.error("找不到 baseline_result.txt，请先运行 exp1_compare.py")
        return
    with open(PCAP_COL_NAME+"_baseline_result.txt", "r", encoding="utf-8") as f:
        query_text = f.read()
    
    # 防止 query 过长，取前 1000 个字符用于检索
    query_text = query_text[:1000]

    client = MongoClient(MONGO_URI)
    db = client[HISTORY_DB_NAME]
    col = db["xlab_threat_intel"]

    logging.info("1. 正在加载所有 XLab 情报库文档...")
    docs = list(col.find({}, {"_id": 1, "title": 1, "url": 1, "content": 1}))
    if not docs:
        logging.error("知识库为空！")
        return

    logging.info(f"加载了 {len(docs)} 篇文档，正在进行分词和向量化计算 (首次运行可能稍慢)...")
    
    # 准备语料
    corpus_tokens = []
    contents = []
    for d in docs:
        corpus_tokens.append(_tokenize(d["content"]))
        contents.append(d["content"])

    # 批量计算知识库向量
    doc_vectors_raw = embeddings.embed_documents(contents)
    doc_vectors = [np.array(vec, dtype=np.float32) for vec in doc_vectors_raw]

    logging.info("2. 正在处理 Query 检索...")
    query_tokens = _tokenize(query_text)
    query_vec = np.array(embeddings.embed_query(query_text), dtype=np.float32)

    # --- 路线 1：BM25 打分 ---
    bm25 = BM25Okapi(corpus_tokens)
    bm25_raw = bm25.get_scores(query_tokens)
    bm25_scores = sorted([(i, float(bm25_raw[i])) for i in range(len(docs))], key=lambda x: x[1], reverse=True)

    # --- 路线 2：向量相似度打分 ---
    vec_scores = sorted([(i, _cosine_similarity(query_vec, doc_vectors[i])) for i in range(len(docs))], key=lambda x: x[1], reverse=True)

    # --- 路线 3：RRF 融合 ---
    rrf_k = 60
    rrf = {}
    for rank, (doc_idx, _) in enumerate(vec_scores):
        rrf[doc_idx] = rrf.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (doc_idx, _) in enumerate(bm25_scores):
        rrf[doc_idx] = rrf.get(doc_idx, 0.0) + 1.0 / (rrf_k + rank + 1)

    final_ranking = sorted(rrf.items(), key=lambda x: x[1], reverse=True)

    # 输出结果
    print("\n" + "="*30 + " 检索融合排名结果 (RRF) " + "="*30)
    target_rank = -1
    for rank, (doc_idx, rrf_score) in enumerate(final_ranking):
        doc = docs[doc_idx]
        if TARGET_TITLE in doc["title"]:
            target_rank = rank + 1
            print(f"🌟 [第 {rank+1} 名] (命中目标!) RRF得分: {rrf_score:.4f} | 标题: {doc['title']}")
        elif rank < 5:  # 打印前 5 名
            print(f"   [第 {rank+1} 名] RRF得分: {rrf_score:.4f} | 标题: {doc['title']}")

    print("\n" + "="*84)
    if target_rank != -1:
        logging.info(f"🎉 实验成功！目标文章排在第 {target_rank} 位 (共 {len(docs)} 篇文章)。")
    else:
        logging.warning("⚠️ 未在检索结果中找到目标文章，请检查标题是否完全匹配。")

if __name__ == "__main__":
    main()