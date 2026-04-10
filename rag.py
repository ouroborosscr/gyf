"""
rag.py — GYF RAG Processing Pipeline (Neo4j 版)
=================================================

整体架构（重构后）：

    【图】= 纯管道，只处理一条文档，没有循环：
        START → extract_think → save_think → semantic_search
              → flow_matching → retrieve_context
              → update_knowledge_graph → final_judgment
              → save_result → END

    【控制器】= 图外面，负责 "喂文档" + 循环 + 错误处理：
        - process_one(doc)   → graph.py 实时调用（流水线模式）
        - process_next()     → 从 MongoDB 取下一条未处理的
        - run(max_docs)      → 循环调 process_next（批量补跑 / 测试用）

    为什么这样设计：
        旧版把循环放在图里（save_result → fetch_doc），一次 invoke 处理所有文档。
        但 graph.py 每产出一条就要立刻处理，不能等全部产出完再批量跑。
        所以改成图只管 "一条进 → 一个判断出"，循环/触发由外面决定。

graph.py 未来集成（只需两行）：
    from rag import rag_processor
    rag_processor.process_one(doc)
"""

# ─────────────────────────── 导入依赖 ───────────────────────────

import re                                       # 正则：提取 <think> 标签、中文分词
import json                                     # JSON：流记录存 Neo4j 时序列化
import logging                                  # 日志
import time                                     # 计时、重试等待
import traceback                                # 打印异常堆栈
from datetime import datetime                   # 时间戳
from typing import Optional                     # 类型注解

import numpy as np                              # 向量余弦相似度计算
from pymongo import MongoClient                 # MongoDB 客户端
from pymongo.errors import ConnectionFailure    # MongoDB 连接异常
from rank_bm25 import BM25Okapi                 # BM25：基于词频的关键词检索
from sentence_transformers import SentenceTransformer  # 文本→384维向量的编码器（~120MB小模型）
from neo4j import GraphDatabase                 # Neo4j 驱动
from neo4j.exceptions import ServiceUnavailable, AuthError  # Neo4j 异常
from langgraph.graph import StateGraph, START, END  # LangGraph 工作流图框架
from typing_extensions import TypedDict         # 类型化字典

from llm import llm, embeddings  # 把 embeddings 一起导进来                             # LLM 实例（llm.py，指向 vLLM 的 Qwen3）
from utils import config                        # 项目配置（MongoDB URI 等）

import jieba
import re



# ─────────────────────────── 配置 ───────────────────────────
logging.basicConfig(                            # 日志格式：时间 - [RAG] - 级别 - 消息
    level=logging.INFO,
    format="%(asctime)s - [RAG] - %(levelname)s - %(message)s",
)

TARGET_DATE = "3_1"                             # 数据集日期，对应集合名 test_status_3_1 和 3_1_conn
TOP_K       = 3                                 # 语义检索和流匹配各取 Top-3

# ⚠️ embedding 模型：把中文文本变成 384 维向量用于语义搜索
# 无法联网时改成本地路径，如 "/date/sunchengrui/models/paraphrase-multilingual-MiniLM-L12-v2"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

NEO4J_URI      = "bolt://localhost:7865"        # Neo4j Bolt 端口（docker-compose 的 neo4j-gyf）
NEO4J_USER     = "neo4j"                        # Neo4j 用户名
NEO4J_PASSWORD = "gyf_password"                 # Neo4j 密码

LLM_MAX_RETRIES = 3                             # LLM 调用最多重试 3 次
LLM_RETRY_DELAY = 5                             # 每次重试间隔 5 秒

# ─────────────────────────── State（一条文档的处理状态） ───────────────────────────
# 没有 processed_count/max_docs/should_stop，那些是控制器的事

class RAGState(TypedDict):
    current_doc:      dict              # 当前处理的 test_status 文档（控制器传入）
    from_batch_skip:  int               # 该批次在 conn 集合中的起始偏移
    batch_limit:      int               # 该批次包含多少条流（通常 30）
    think_summary:    str               # Node1 输出：LLM 整理的思考摘要
    think_embedding:  list              # Node2 输出：摘要的 384 维向量
    semantic_uids:    list              # Node3 输出：语义检索 Top-3 批次编号
    flow_uids:        list              # Node4 输出：流匹配 Top-3 流的 uid
    all_context_docs: list              # Node5 输出：去重后的历史参考文档
    graph_context:    str               # Node6 输出：Neo4j 图谱上下文文本
    final_judgment:   str               # Node7 输出：LLM 最终安全判断

# ─────────────────────────── 全局资源（懒加载单例） ───────────────────────────
# 首次使用时创建，之后复用，不用每条文档都重连

_embed_model:  Optional[SentenceTransformer] = None   # 向量化模型
_mongo_client: Optional[MongoClient]         = None   # MongoDB 连接
_neo4j_driver = None                                  # Neo4j 驱动
_history_db   = None                                  # gyf_history 库
_zeek_db      = None                                  # zeek 库

def _get_embed_model() -> SentenceTransformer:
    """懒加载向量化模型，首次调用时加载到 GPU"""
    global _embed_model
    if _embed_model is None:
        logging.info(f"加载嵌入模型: {EMBED_MODEL}")
        _embed_model = SentenceTransformer(EMBED_MODEL)      # 自动用 GPU
    return _embed_model

def _get_mongo():
    """懒加载 MongoDB，返回 (history_db, zeek_db)"""
    global _mongo_client, _history_db, _zeek_db
    if _mongo_client is None:
        uri           = config.DATABASE["mongo"]["uri"]      # 从 config.py 读连接串
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)  # 5秒超时
        _history_db   = _mongo_client[config.GYF_SETTINGS.get("history_db_name", "gyf_history")]
        _zeek_db      = _mongo_client[config.DATABASE["mongo"]["db_name"]]
    return _history_db, _zeek_db

def _get_neo4j():
    """懒加载 Neo4j，首次连接时创建 Host.ip 唯一约束"""
    global _neo4j_driver
    if _neo4j_driver is None:
        logging.info(f"连接 Neo4j: {NEO4J_URI}")
        _neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with _neo4j_driver.session() as session:             # 幂等：重复执行不报错
            session.run("CREATE CONSTRAINT host_ip_unique IF NOT EXISTS FOR (h:Host) REQUIRE h.ip IS UNIQUE")
    return _neo4j_driver

# ─────────────────────────── 工具函数 ───────────────────────────

def _extract_think_text(raw_output: str) -> str:
    """从 Qwen3 输出中提取 <think>思考内容</think>，没有标签则返回全文"""
    if not raw_output or not raw_output.strip():             # 空值保护
        return ""
    match = re.search(r"<think>(.*?)</think>", raw_output, re.DOTALL)
    return match.group(1).strip() if match else raw_output

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """余弦相似度 cos(θ)=a·b/(|a|×|b|)，范围[-1,1]，越大越相似"""
    if a.size == 0 or b.size == 0 or a.shape != b.shape:    # 空/维度不匹配→0
        return 0.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)           # L2范数
    if na == 0 or nb == 0:                                   # 零向量→0
        return 0.0
    return float(np.dot(a, b) / (na * nb))

# 提前注册安全领域的专有名词，防止被 jieba 误切
# 比如防止 "内网穿透" 被切成 "内网" 和 "穿透"
SECURITY_KEYWORDS = [
    "内网穿透", "隐蔽信道", "端口扫描", "反弹shell", "提权",
    "加密通信", "横向移动", "暴力破解", "注入攻击", "恶意载荷"
]
for kw in SECURITY_KEYWORDS:
    jieba.add_word(kw)

def _tokenize(text: str) -> list:
    """
    使用 jieba 进行中英文混合分词（专为 BM25 优化）。
    """
    if not text or not text.strip():
        return ["__empty__"]
    
    # 1. 使用 jieba 精确模式进行切词
    raw_tokens = jieba.lcut(text)
    
    result = []
    for t in raw_tokens:
        t = t.strip().lower() # 转小写，去首尾空格
        # 2. 过滤掉无意义的单字符标点符号、停用词或纯空白
        if t and not re.match(r'^[^\w\s\u4e00-\u9fff]+$', t):
            result.append(t)
            
    return result if result else ["__empty__"]

def _safe_float(v) -> float:
    """任意值→float，None/异常→0.0"""
    try: return float(v) if v is not None else 0.0
    except (TypeError, ValueError): return 0.0

def _build_batch_profile(flows: list) -> np.ndarray:
    """
    从一批流中提取「统计特征画像」向量，用于批次间余弦相似度比较。

    维度设计（共 38 维）：

    ┌─ 1. 数值类特征：duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts
    │     每个取 [max, min, median, std]
    │     → 5 × 4 = 20 维
    │
    ├─ 2. 端口分箱比例（源端口 + 目的端口各一组）
    │     分别统计：
    │       - 知名端口 (0-1023) 占比
    │       - 注册端口 (1024-49151) 占比
    │       - 动态/私有端口 (>49151) 占比
    │       - 高危端口命中率 (22,23,25,53,80,135,139,443,445,993,995,
    │                         1433,1521,3306,3389,4444,5432,5900,6379,8080,8443,8888,9200)
    │     → 2 × 4 = 8 维
    │
    ├─ 3. 网络行为指标
    │       - 源IP 多样性, 目的IP 多样性, 流量方向比, TCP 占比, UDP 占比
    │     → 5 维
    │
    └─ 4. 端口多样性指标
          - 目的端口多样性, 源端口多样性, 最常见目的端口占比, 最常见源端口占比, DNS(53)占比
        → 5 维

    总计 20 + 8 + 5 + 5 = 38 维，L2 归一化。
    """
    from collections import Counter

    n = len(flows)
    if n == 0:
        return np.zeros(38, dtype=np.float32)

    # 常见高危/敏感端口集合
    HIGH_RISK_PORTS = {
        22, 23, 25, 53, 80, 135, 139, 443, 445, 993, 995,  # 基础服务
        1433, 1521, 3306, 3389,                              # 数据库 + RDP
        4444, 5432, 5900, 6379,                              # 反弹shell / PostgreSQL / VNC / Redis
        8080, 8443, 8888, 9200,                              # Web 服务 / Elasticsearch
    }

    stats = []

    # ── 1. 数值类特征 × [max, min, median, std]（5×4=20维）──
    numeric_keys = ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]
    for key in numeric_keys:
        vals = np.array([_safe_float(f.get(key)) for f in flows], dtype=np.float32)
        if len(vals) > 0:
            stats.extend([
                float(np.max(vals)),                             # 最大值：检测异常峰值
                float(np.min(vals)),                             # 最小值：检测异常低值
                float(np.median(vals)),                          # 中位数：比均值更抗极端值
                float(np.std(vals)),                             # 标准差：波动程度
            ])
        else:
            stats.extend([0.0, 0.0, 0.0, 0.0])
    
    dst_ports = [int(_safe_float(f.get("id.resp_p"))) for f in flows]
    src_ports = [int(_safe_float(f.get("id.orig_p"))) for f in flows]

    # ── 2. (新增) 端口数值统计 × [max, min, median, std]（2×4=8维）──
    for ports in [dst_ports, src_ports]:
        if ports:
            stats.extend([
                float(np.max(ports)), float(np.min(ports)), 
                float(np.median(ports)), float(np.std(ports))
            ])
        else:
            stats.extend([0.0, 0.0, 0.0, 0.0])

    # ── 2. 端口分箱比例（源端口4维 + 目的端口4维 = 8维）──
    def _port_bins(ports: list) -> list:
        """端口列表 → [知名比, 注册比, 动态比, 高危命中率]"""
        well_known = sum(1 for p in ports if 0 <= p <= 1023)
        registered = sum(1 for p in ports if 1024 <= p <= 49151)
        dynamic    = sum(1 for p in ports if p > 49151)
        high_risk  = sum(1 for p in ports if p in HIGH_RISK_PORTS)
        return [well_known / n, registered / n, dynamic / n, high_risk / n]
        
    stats.extend(_port_bins(dst_ports))                      # [20-23] 目的端口分箱
    stats.extend(_port_bins(src_ports))                      # [24-27] 源端口分箱

    # ── 3. 网络行为指标（5维）──
    unique_src_ips = set(f.get("id.orig_h", "") for f in flows)
    unique_dst_ips = set(f.get("id.resp_h", "") for f in flows)
    total_orig = sum(_safe_float(f.get("orig_bytes")) for f in flows)
    total_resp = sum(_safe_float(f.get("resp_bytes")) for f in flows)
    protos     = [f.get("proto", "").lower() for f in flows]

    stats.extend([
        len(unique_src_ips) / n,                             # [28] 源IP多样性
        len(unique_dst_ips) / n,                             # [29] 目的IP多样性
        total_orig / (total_orig + total_resp + 1),          # [30] 流量方向比（1=纯上传）
        sum(1 for p in protos if p == "tcp") / n,            # [31] TCP 占比
        sum(1 for p in protos if p == "udp") / n,            # [32] UDP 占比
    ])

    # ── 4. 端口多样性指标（5维）──
    dst_counter = Counter(dst_ports)
    src_counter = Counter(src_ports)
    stats.extend([
        len(set(dst_ports)) / n,                             # [33] 目的端口多样性（扫描→1）
        len(set(src_ports)) / n,                             # [34] 源端口多样性
        dst_counter.most_common(1)[0][1] / n if dst_counter else 0,  # [35] 最常见目的端口占比
        src_counter.most_common(1)[0][1] / n if src_counter else 0,  # [36] 最常见源端口占比
        sum(1 for p in dst_ports if p == 53) / n,            # [37] DNS(53) 占比
    ])

    vec = np.array(stats, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec                   # L2 归一化

def _extract_payload_summary(flows: list) -> str:
    """
    提取流中的关键 Payload 信息并去重拼接。
    截取头部以适应 embedding 模型的最大 Token 限制。
    """
    payloads = set()
    for f in flows:
        p = f.get("stream_payload_decoded", "").strip()
        if p:
            # 取每条流的前 256 个字符（覆盖 HTTP 头、SQLi 语句或命令前缀）
            payloads.add(p[:256])
    
    if not payloads:
        return ""
    # 拼接去重后的 payload，总长度限制在 1500 字符内
    return "\n---\n".join(list(payloads))[:1500]

def _invoke_llm_with_retry(prompt: str, retries: int = LLM_MAX_RETRIES) -> str:
    """调用 LLM 带重试：发请求→提取文本→去<think>标签→失败等5s重试→3次全败返降级文本"""
    for attempt in range(1, retries + 1):
        try:
            response = llm.invoke(prompt)                    # HTTP POST → vLLM
            text = response.content if hasattr(response, "content") else str(response)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()  # 去思考标签
            return text
        except Exception as e:
            logging.warning(f"⚠️ LLM 调用失败 (尝试 {attempt}/{retries}): {e}")
            if attempt < retries:
                logging.info(f"   {LLM_RETRY_DELAY}s 后重试…")
                time.sleep(LLM_RETRY_DELAY)
            else:
                logging.error(f"❌ LLM 调用彻底失败，返回降级文本")
                return f"[LLM 调用失败] 错误: {str(e)}"      # 不崩溃，管道继续

# ─────────────────────────── 启动检查 ───────────────────────────

def _check_connectivity():
    """启动前检查 MongoDB + 集合 + Neo4j，任何不通直接退出"""
    try:                                                     # — MongoDB —
        history_db, zeek_db = _get_mongo()
        _mongo_client.admin.command("ping")                  # ping 检测
        logging.info("✅ MongoDB 连接正常")
    except (ConnectionFailure, Exception) as e:
        logging.error(f"❌ MongoDB 连接失败: {e}"); raise SystemExit(1)

    status_col_name = f"test_status_{TARGET_DATE}"           # — GYF_Test 输出集合 —
    if status_col_name not in history_db.list_collection_names():
        logging.error(f"❌ 集合 {status_col_name} 不存在，请先运行 GYF_Test.py"); raise SystemExit(1)
    unprocessed = history_db[status_col_name].count_documents(
        {"from_batch_skip": {"$exists": True}, "rag_processed": {"$ne": True}})
    logging.info(f"📋 待处理文档数: {unprocessed}")

    conn_col_name = f"{TARGET_DATE}_conn"                    # — Zeek 流量集合 —
    if conn_col_name not in zeek_db.list_collection_names():
        logging.error(f"❌ 集合 {conn_col_name} 不存在"); raise SystemExit(1)
    logging.info(f"📋 流量总数: {zeek_db[conn_col_name].count_documents({})}")

    try:                                                     # — Neo4j —
        driver = _get_neo4j(); driver.verify_connectivity()
        logging.info("✅ Neo4j 连接正常")
    except (ServiceUnavailable, AuthError, Exception) as e:
        logging.error(f"❌ Neo4j 连接失败: {e}")
        logging.error("   请确认: docker compose up -d neo4j-gyf && 等待 15s"); raise SystemExit(1)


# ╔══════════════════════════════════════════════════════════════╗
# ║         图的 8 个节点（纯管道，处理一条文档）                    ║
# ║   START → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → END            ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Node 1: extract_think ──
# 作用：从 model_raw_output 提取 <think> 内容，让 LLM 整理成结构化摘要
# 输出：think_summary → 传给 Node2

def extract_think(state: RAGState) -> RAGState:
    doc  = state["current_doc"]                              # 当前文档
    x, y = state["from_batch_skip"], state["from_batch_skip"] + state["batch_limit"] - 1  # 流量范围
    think_text = _extract_think_text(doc.get("model_raw_output", ""))  # 提取思考内容

    if not think_text:                                       # 空内容保护（normal_log 可能没有）
        logging.warning(f"⚠️ 文档 skip={x} 无 model_raw_output，使用兜底文本")
        fallback = f"本批次（流 {x}~{y}）类型为 {doc.get('type','unknown')}，无详细思考过程记录。"
        return {**state, "think_summary": fallback}          # 跳过 LLM，直接用兜底

    prompt = (                                               # 构建提示词
        f"异常流量是第 {x} 到第 {y} 条，"
        f"帮我整理以下思考过程中对第 {x} 到第 {y} 条的思考内容。\n\n"
        f"思考过程：\n{think_text}")
    logging.info(f"🧠 LLM 整理思考内容 (流 {x}~{y})…")
    summary = _invoke_llm_with_retry(prompt)                 # 调用 LLM（带重试）
    logging.info(f"✅ 思考内容提取完成 (长度={len(summary)})")
    return {**state, "think_summary": summary}

# ── Node 2: save_think ──
# 作用：摘要→向量化 + 提取流特征向量 → 存入 rag_knowledge_base（知识库的"写"操作）
# 数据来源：doc["suspicious_flows"]（test_status 文档里已经存了完整流对象，不需要查 conn）
# 输出：think_embedding → 传给 Node3

def save_think(state: RAGState) -> RAGState:
    history_db, _ = _get_mongo()
    kb_col = history_db["rag_knowledge_base"] # RAG 知识库

    doc = state["current_doc"]
    flows = doc.get("suspicious_flows", [])

    # —— 1. 处理 LLM 思考摘要向量 (使用 LangChain 接入的 4B 模型) ——
    summary = state["think_summary"] # 上一步的摘要
    if summary:
        think_vec_raw = embeddings.embed_query(summary)
        think_vec_np = np.array(think_vec_raw, dtype=np.float32)
        think_vec = (think_vec_np / np.linalg.norm(think_vec_np)).tolist() # L2 归一化
    else:
        think_vec = []

    # —— 2. 提取并归一化 46 维网络行为统计特征 ——
    # _build_batch_profile 内部已做 L2 归一化，此时 stats_vec 长度为 1
    stats_vec = _build_batch_profile(flows)

    # —— 3. 提取 Payload 并转化为大模型向量 ——
    payload_text = _extract_payload_summary(flows)
    if payload_text:
        payload_vec_raw = embeddings.embed_query(payload_text)
        payload_vec = np.array(payload_vec_raw, dtype=np.float32)
        payload_vec = payload_vec / np.linalg.norm(payload_vec) # L2 归一化
    else:
        # ⚠️ 适配 4B 模型：Qwen3-Embedding-4B 输出维度为 2560 维（若你实测不同请修改此数值）
        payload_vec = np.zeros(2560, dtype=np.float32)
    
    # ✅ 提前分词，空间换时间
    # 遵照你之前的架构要求，只对 LLM 摘要进行分词，避免混淆代码/特征码污染 BM25 词袋
    doc_tokens = _tokenize(summary)

    # —— 4. 权重控制与拼接 (核心防淹没逻辑) ——
    # 设定特征重要性：比如行为占 60%，Payload 占 40%
    WEIGHT_STATS = 0.6
    WEIGHT_PAYLOAD = 0.4
    
    # 乘以权重的平方根，拼接后的 2606 维向量 (46 + 2560) 的 L2 范数会自然等于 1
    combined_vec = np.concatenate([
        np.sqrt(WEIGHT_STATS) * stats_vec,
        np.sqrt(WEIGHT_PAYLOAD) * payload_vec
    ])
    
    # 兜底保护：如果某部分全零导致长度微小偏差，重新归一化一次
    norm = np.linalg.norm(combined_vec)
    combined_vec = combined_vec / norm if norm > 0 else combined_vec
    flow_agg_vec = combined_vec.tolist()

    suspicious_uids = [f["uid"] for f in flows if "uid" in f] # 提取所有流的 uid
    first_flow_uid  = suspicious_uids[0] if suspicious_uids else "" # 第一条流的uid，供其他节点用

    kb_doc = { # 组装知识库文档
        "source_doc_id":        str(doc["_id"]), # 来源文档 ID
        "from_batch_skip":      state["from_batch_skip"], # 批次编号（唯一键）
        "batch_limit":          state["batch_limit"], # 批次大小
        "doc_type":             doc.get("type", "unknown"), # normal_log 或 suspicious_alert
        "think_summary":        summary, # 思考摘要文本
        "think_embedding":      think_vec, # Qwen3-4B 的高维语义向量
        "payload_text":         payload_text,                # 存明文，供 Node 7 (LLM) 参考
        "doc_tokens":           doc_tokens,  # ✅ 将分词数组存入 MongoDB
        "flow_feature_vec":     flow_agg_vec,                # 现在的带权 2606 维融合指纹
        "suspicious_flow_uids": suspicious_uids, # 所有可疑流的uid列表
        "first_flow_uid":       first_flow_uid, # 第一条流uid（供 flow_matching 用）
        "batch_flows":          flows, # 完整流对象（供 retrieve_context 用）
        "updated_at":           datetime.now(), # 时间戳
    }
    
    # upsert：存在更新，不存在插入（幂等）
    kb_col.update_one({"from_batch_skip": state["from_batch_skip"]}, {"$set": kb_doc}, upsert=True) 
    # 顺便把摘要回写到源文档
    history_db[f"test_status_{TARGET_DATE}"].update_one({"_id": doc["_id"]}, {"$set": {"think_summary": summary}}) 
    
    logging.info(f"💾 知识库写入: skip={state['from_batch_skip']}，生成带权流特征 (行为 {WEIGHT_STATS} : 载荷 {WEIGHT_PAYLOAD})")
    
    return {**state, "think_embedding": think_vec} # 向量传给 Node3

# ── Node 3: semantic_search ──
# 作用：用向量+BM25双路检索历史知识库，RRF融合取Top-3
# 向量路：捕捉"意思相近"  BM25路：捕捉"关键词相同"
# 输出：semantic_uids → 传给 Node5

def semantic_search(state: RAGState) -> RAGState:
    history_db, _ = _get_mongo()
    kb_col = history_db["rag_knowledge_base"]

    current_skip    = state["from_batch_skip"]               # 当前批次编号（排除自己）
    current_vec     = np.array(state.get("think_embedding", []), dtype=np.float32)  # 当前向量
    current_summary = state["think_summary"]                 # 当前摘要

    if current_vec.size == 0:                                # 空向量保护
        logging.warning("⚠️ 当前文档无有效 embedding，跳过语义检索")
        return {**state, "semantic_uids": []}

    # ✅ 优化点 1：查询时，直接拉取数据库里存好的 doc_tokens 数组，不再拉取 think_summary 文本
    past_docs = list(kb_col.find(                            
        {"from_batch_skip": {"$ne": current_skip}, "think_embedding": {"$exists": True, "$ne": []}},
        {"from_batch_skip": 1, "think_embedding": 1, "doc_tokens": 1}))
        
    if not past_docs:                                        # 第一条时没有历史
        logging.info("🔍 无历史知识库文档，跳过语义检索。")
        return {**state, "semantic_uids": []}

    # —— 路线1：向量余弦相似度 ——
    vec_scores = sorted(                                     # 逐条算相似度，降序排
        [(d["from_batch_skip"],
          _cosine_similarity(current_vec, np.array(d.get("think_embedding", []), dtype=np.float32)))
         for d in past_docs], key=lambda x: x[1], reverse=True)

    # —— 路线2：BM25关键词 ——
    # ✅ 优化点 2：直接使用预先分好的数组构建语料库，彻底干掉 CPU 密集的循环分词！
    corpus       = [d.get("doc_tokens", ["__empty__"]) for d in past_docs]  
    
    bm25         = BM25Okapi(corpus)                         # 建BM25索引
    query_tokens = _tokenize(current_summary)                # 只有当前文档需要即时分词 1 次
    bm25_raw     = bm25.get_scores(query_tokens)             # 算BM25得分
    
    bm25_scores  = sorted(
        [(past_docs[i]["from_batch_skip"], float(bm25_raw[i])) for i in range(len(past_docs))],
        key=lambda x: x[1], reverse=True)

    # —— RRF融合：两路排名加权合并 ——
    # 公式 RRF(d) = Σ 1/(60+rank)，两路都靠前的得分最高
    rrf_k, rrf = 60, {}
    for rank, (uid, _) in enumerate(vec_scores):             # 向量排名贡献
        rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, (uid, _) in enumerate(bm25_scores):            # BM25排名贡献
        rrf[uid] = rrf.get(uid, 0.0) + 1.0 / (rrf_k + rank + 1)

    top_uids = [uid for uid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]
    logging.info(f"🔍 语义检索 Top-{TOP_K}: {top_uids}")
    
    return {**state, "semantic_uids": top_uids}

# ── Node 4: flow_matching ──
# 作用：用9维流特征向量找历史上流量特征最像的批次（和Node3互补）
# Node3比文本语义，Node4比数值特征
# 输出：flow_uids → 传给 Node5

def flow_matching(state: RAGState) -> RAGState:
    history_db, _ = _get_mongo()
    kb_col = history_db["rag_knowledge_base"]

    kb_cur = kb_col.find_one({"from_batch_skip": state["from_batch_skip"]}, {"flow_feature_vec": 1})
    if not kb_cur or not kb_cur.get("flow_feature_vec"):
        logging.warning("⚠️ 当前批次未找到 flow_feature_vec，跳过流匹配。")
        return {**state, "flow_uids": []}

    cur_vec = np.array(kb_cur["flow_feature_vec"], dtype=np.float32)  # 完整的 430 维融合向量

    past_docs = list(kb_col.find(                            
        {"from_batch_skip": {"$ne": state["from_batch_skip"]}, "flow_feature_vec": {"$exists": True}},
        {"from_batch_skip": 1, "flow_feature_vec": 1, "suspicious_flow_uids": 1, "first_flow_uid": 1}))
        
    if not past_docs:
        logging.info("🔄 无历史流特征向量，跳过流匹配。")
        return {**state, "flow_uids": []}

    COARSE_TOP_N = 100  # 粗排候选集大小（可根据你的库总数据量调整）

    # ==========================================
    # 阶段 1：粗排 (Coarse Ranking) - 极速筛选
    # 策略：只取前 46 维（纯网络行为特征）算相似度
    # ==========================================
    cur_coarse_vec = cur_vec[:46]  # 向量切片提取粗排特征
    
    coarse_scores = []
    for d in past_docs:
        past_vec = np.array(d["flow_feature_vec"], dtype=np.float32)
        past_coarse_vec = past_vec[:46]
        
        # 粗排只算 46 维的点积，CPU 开销极小
        sim = _cosine_similarity(cur_coarse_vec, past_coarse_vec)
        coarse_scores.append((sim, d, past_vec))
        
    # 按粗排相似度降序，截取 Top-N 进入精排池
    coarse_scores.sort(key=lambda x: x[0], reverse=True)
    coarse_candidates = coarse_scores[:COARSE_TOP_N]

    # ==========================================
    # 阶段 2：精排 (Fine Ranking) - 精准打击
    # 策略：在候选集里，使用完整的 430 维（加入 Payload 语义）算相似度
    # ==========================================
    fine_scores = []
    for coarse_sim, d, past_vec in coarse_candidates:
        # 精排计算完整的 430 维相似度
        final_sim = _cosine_similarity(cur_vec, past_vec)
        fine_scores.append((final_sim, d))
        
    # 按精排最终得分降序，取最终的 Top-K (默认 3)
    fine_scores.sort(key=lambda x: x[0], reverse=True)
    top_docs = fine_scores[:TOP_K]

    # ==========================================
    # 阶段 3：提取代表性流 UID 传递给下一节点
    # ==========================================
    flow_uids = []
    for sim_score, doc in top_docs:
        susp = doc.get("suspicious_flow_uids", [])
        if susp:
            flow_uids.append(susp[0])
        else:
            fuid = doc.get("first_flow_uid", "")
            if fuid:
                flow_uids.append(fuid)
                
    logging.info(f"🔄 流特征匹配 (粗排取Top{COARSE_TOP_N} -> 精排取Top{TOP_K}): {flow_uids}")
    
    return {**state, "flow_uids": flow_uids}

# ── Node 5: retrieve_context ──
# 作用：合并Node3+Node4的结果（去重），从MongoDB取完整流内容和思考摘要
# 这些历史参考会作为提示词的一部分喂给LLM做最终判断
# 输出：all_context_docs → 传给 Node7

def retrieve_context(state: RAGState) -> RAGState:
    history_db, _ = _get_mongo()
    kb_col = history_db["rag_knowledge_base"]
    seen, context_docs = set(), []                           # seen：去重用

    def _add_by_skip(skip_val):                              # 按批次编号取（语义检索的结果）
        kb_doc = kb_col.find_one({"from_batch_skip": skip_val})
        if not kb_doc: return
        susp     = kb_doc.get("suspicious_flow_uids", [])
        flow_uid = susp[0] if susp else kb_doc.get("first_flow_uid", "")  # 用 kb 里存的 uid
        if not flow_uid or flow_uid in seen: return
        seen.add(flow_uid)
        # 从 kb 存的 batch_flows 里找这条流的完整内容（不查 conn）
        flow_content = {}
        for f in kb_doc.get("batch_flows", []):
            if f.get("uid") == flow_uid:
                flow_content = f; break
        context_docs.append({
            "uid":           flow_uid,
            "flow_content":  flow_content,                   # 从 kb 取，不查 conn
            "think_summary": kb_doc.get("think_summary", "")})

    def _add_by_flow_uid(flow_uid: str):                     # 按流uid取（流匹配的结果）
        if flow_uid in seen: return
        seen.add(flow_uid)
        # 在所有 kb 文档的 batch_flows 里搜索包含此 uid 的流（不查 conn）
        kb_doc = kb_col.find_one({"batch_flows.uid": flow_uid})
        if not kb_doc: return
        flow_content = {}
        for f in kb_doc.get("batch_flows", []):
            if f.get("uid") == flow_uid:
                flow_content = f; break
        context_docs.append({
            "uid": flow_uid, "flow_content": flow_content,
            "think_summary": kb_doc.get("think_summary", "")})

    for s in state["semantic_uids"]:  _add_by_skip(s)        # 处理语义检索结果
    for u in state["flow_uids"]:      _add_by_flow_uid(u)    # 处理流匹配结果
    logging.info(f"📦 检索到上下文文档 {len(context_docs)} 条")
    return {**state, "all_context_docs": context_docs}

# ── Node 6: update_knowledge_graph ──
# 作用：写入 Neo4j（Host节点+CONNECTED_TO边）然后读取图谱信息
# Neo4j 挂了不崩溃，降级为空上下文
# 输出：graph_context → 传给 Node7

def update_knowledge_graph(state: RAGState) -> RAGState:
    try:
        return _do_update_knowledge_graph(state)
    except (ServiceUnavailable, Exception) as e:             # Neo4j 挂了→降级
        logging.error(f"❌ Neo4j 操作失败，降级跳过图谱: {e}")
        return {**state, "graph_context": "（Neo4j 连接异常，图谱上下文不可用）"}

def _do_update_knowledge_graph(state: RAGState) -> RAGState:
    driver = _get_neo4j()
    # 直接从当前文档取流（不查 conn）
    flows = state["current_doc"].get("suspicious_flows", [])
    involved_hosts: set = set()

    with driver.session() as session:                        # ══ 写入阶段 ══
        for flow in flows:
            src = flow.get("id.orig_h")                      # 源IP
            dst = flow.get("id.resp_h")                      # 目的IP
            if not src or not dst: continue                  # 缺IP跳过
            orig_pkts = int(flow.get("orig_pkts") or 0)      # 源端包数
            resp_pkts = int(flow.get("resp_pkts") or 0)      # 目的端包数
            flow_rec  = json.dumps({                         # 流详情序列化成JSON串存到边上
                "uid": flow.get("uid",""), "src_port": flow.get("id.orig_p"),
                "dst_port": flow.get("id.resp_p"), "duration": flow.get("duration"),
                "orig_bytes": flow.get("orig_bytes"), "resp_bytes": flow.get("resp_bytes"),
            }, ensure_ascii=False)

            session.run("""                                  
                MERGE (s:Host {ip: $src})                    
                ON CREATE SET s.send_pkts = $op, s.recv_pkts = 0
                ON MATCH  SET s.send_pkts = s.send_pkts + $op
                MERGE (d:Host {ip: $dst})                    
                ON CREATE SET d.recv_pkts = $rp, d.send_pkts = 0
                ON MATCH  SET d.recv_pkts = d.recv_pkts + $rp
            """, src=src, dst=dst, op=orig_pkts, rp=resp_pkts)  # MERGE节点+累加包数

            session.run("""                                  
                MATCH (s:Host {ip: $src})
                MATCH (d:Host {ip: $dst})
                MERGE (s)-[r:CONNECTED_TO]->(d)              
                ON CREATE SET r.flow_count = 1, r.flows = [$flow_rec]
                ON MATCH  SET r.flow_count = r.flow_count + 1,
                              r.flows      = r.flows + [$flow_rec]
            """, src=src, dst=dst, flow_rec=flow_rec)        # MERGE边+追加流记录
            involved_hosts.update([src, dst])
    logging.info(f"🕸️  Neo4j 知识图谱更新完成，涉及主机 {len(involved_hosts)} 个")

    lines = []                                               # ══ 读取阶段：构建上下文 ══
    with driver.session() as session:
        for host in involved_hosts:
            node_res = session.run(                          # 查节点累计收发包
                "MATCH (h:Host {ip: $ip}) RETURN h.send_pkts AS sp, h.recv_pkts AS rp", ip=host).single()
            if node_res:
                lines.append(f"[主机节点] {host}: 累计发包={node_res['sp']}, 累计收包={node_res['rp']}")
            for rec in session.run(                          # 查出边（该IP连了谁）
                "MATCH (s:Host {ip: $ip})-[r:CONNECTED_TO]->(d:Host) "
                "RETURN d.ip AS dst, r.flow_count AS fc, r.flows AS flows", ip=host):
                last3 = rec["flows"][-3:] if rec["flows"] else []
                lines.append(f"  → {rec['dst']}: 历史流条数={rec['fc']}, 最近3条={last3}")
            for rec in session.run(                          # 查入边（谁连了该IP）
                "MATCH (s:Host)-[r:CONNECTED_TO]->(d:Host {ip: $ip}) "
                "RETURN s.ip AS src, r.flow_count AS fc, r.flows AS flows", ip=host):
                last3 = rec["flows"][-3:] if rec["flows"] else []
                lines.append(f"  ← {rec['src']}: 历史流条数={rec['fc']}, 最近3条={last3}")
    graph_context = "\n".join(lines) if lines else "（当前批次无有效 IP 信息）"
    return {**state, "graph_context": graph_context}

# ── Node 7: final_judgment ──
# 作用：三路信息（当前思考+历史参考+图谱）合并成提示词，让LLM做最终判断
# 这是 RAG 的核心价值：不只看当前批次，还结合了历史和拓扑
# 输出：final_judgment → 传给 Node8

def final_judgment(state: RAGState) -> RAGState:
    x, y = state["from_batch_skip"], state["from_batch_skip"] + state["batch_limit"] - 1
    ctx_lines = []                                           # 格式化历史参考
    for i, item in enumerate(state["all_context_docs"], 1):
        ctx_lines.append(
            f"--- 参考记录 {i} (uid={item['uid']}) ---\n"
            f"流内容: {json.dumps(item['flow_content'], ensure_ascii=False, default=str)}\n"
            f"思考摘要: {item['think_summary']}")
    ctx_str = "\n\n".join(ctx_lines) if ctx_lines else "（无历史参考记录）"

    prompt = f"""你是一名网络安全分析专家，正在对第 {x} 到第 {y} 条流量进行威胁研判。

【当前批次思考摘要】
{state["think_summary"]}

【历史相似案例（语义 + 流特征检索）】
{ctx_str}

【知识图谱上下文（源/目的主机历史行为）】
{state["graph_context"]}

请综合以上信息，判断当前批次流量是否存在威胁（内网穿透/渗透隐蔽信道/加密通信）。

⚠️ 请务必严格按照以下格式输出你的研判结果，不要添加任何多余的修饰词或变体格式：

【判断结论】：（写出明确的结论，如“存在隐蔽信道威胁”或“正常流量”）
【置信度】：（这里只填 0 到 1 之间的小数，例如 0.85）
【关键证据】：（列出支撑你结论的核心证据）
【关联分析】：（结合历史相似案例和图谱上下文进行对比说明）"""

    logging.info(f"⚖️  LLM 最终判断 (流 {x}~{y})…")
    judgment = _invoke_llm_with_retry(prompt)                # 调用LLM
    logging.info(f"✅ 最终判断完成 (长度={len(judgment)})")
    print(f"\n{'='*60}\n[RAG 最终判断 | 流 {x}~{y}]\n{judgment}\n{'='*60}\n")  # 终端实时输出
    return {**state, "final_judgment": judgment}

# ── Node 8: save_result ──
# 作用：写回MongoDB，标记 rag_processed=True（process_next不会再取到这条）

def save_result(state: RAGState) -> RAGState:
    history_db, _ = _get_mongo()
    doc = state["current_doc"]
    judgment_text = state["final_judgment"]
    
    # 🎯 1. 极其健壮的正则提取置信度分数
    score = -1.0  # 默认兜底值
    # 匹配模式：【置信度】 + 任意个冒号(中英文) + 任意个空格 + 数字/小数
    match = re.search(r"【置信度】[：:]\s*([0-9]*\.?[0-9]+)", judgment_text)
    
    if match:
        try:
            score = float(match.group(1))
        except ValueError:
            logging.warning(f"⚠️ 置信度数字转换失败，大模型输出了: {match.group(1)}")
    else:
        # 如果大模型连【置信度】这四个字都没按格式写，我们尝试全局盲抓第一个单独的小数
        fallback_match = re.search(r"(?:置信度).*?([0-9]*\.?[0-9]+)", judgment_text)
        if fallback_match:
            try:
                score = float(fallback_match.group(1))
                logging.info(f"使用盲抓方案提取到分数: {score}")
            except ValueError:
                pass
        else:
            logging.warning("⚠️ 未能从大模型输出中提取到置信度分数！")
            
    # 🎯 2. 写回 test_status (新增 rag_score 字段)
    history_db[f"test_status_{TARGET_DATE}"].update_one(     
        {"_id": doc["_id"]},
        {"$set": {
            "rag_processed":     True,                       # 标记已处理
            "rag_judgment":      judgment_text,              # 判断结果
            "rag_score":         score,                      # 🌟 提取出的纯数字置信度
            "rag_semantic_uids": state["semantic_uids"],     # 语义检索结果
            "rag_flow_uids":     state["flow_uids"],         # 流匹配结果
            "rag_updated_at":    datetime.now()}})           # 处理时间
            
    # 🎯 3. 同步更新知识库 (同时把分数存进去，以后可以按分数排序！)
    history_db["rag_knowledge_base"].update_one(             
        {"from_batch_skip": state["from_batch_skip"]},
        {"$set": {
            "rag_judgment": judgment_text,
            "rag_score":    score                            # 🌟 知识库也存一份
        }})
        
    logging.info(f"💾 结果已保存 (提取得分: {score})，skip={state['from_batch_skip']} 标记为已处理。")
    
    return state                                             # 管道结束

# ─────────────────────────── 构建图（纯管道，无循环） ───────────────────────────

def build_rag_graph():
    builder = StateGraph(RAGState)                           # 创建图
    builder.add_node("extract_think",          extract_think)       # 注册8个节点
    builder.add_node("save_think",             save_think)
    builder.add_node("semantic_search",        semantic_search)
    builder.add_node("flow_matching",          flow_matching)
    builder.add_node("retrieve_context",       retrieve_context)
    builder.add_node("update_knowledge_graph", update_knowledge_graph)
    builder.add_node("final_judgment",         final_judgment)
    builder.add_node("save_result",            save_result)
    builder.add_edge(START,                    "extract_think")     # 串联：纯线性管道
    builder.add_edge("extract_think",          "save_think")
    builder.add_edge("save_think",             "semantic_search")
    builder.add_edge("semantic_search",        "flow_matching")
    builder.add_edge("flow_matching",          "retrieve_context")
    builder.add_edge("retrieve_context",       "update_knowledge_graph")
    builder.add_edge("update_knowledge_graph", "final_judgment")
    builder.add_edge("final_judgment",         "save_result")
    builder.add_edge("save_result",            END)                # 一条处理完就结束
    return builder.compile()

rag_graph = build_rag_graph()                                # 模块加载时构建一次


# ╔══════════════════════════════════════════════════════════════╗
# ║       控制器（图外面：喂文档 + 循环 + 错误处理）                 ║
# ╚══════════════════════════════════════════════════════════════╝

class RAGProcessor:
    """
    三种调用方式：
    1. process_one(doc)  — graph.py 流水线：每产出一条就调
    2. process_next()    — 从MongoDB取下一条未处理的
    3. run(max_docs)     — 批量循环（python rag.py --max-docs 10）
    """
    def __init__(self, target_date: str = TARGET_DATE):
        self.target_date = target_date
        config.GYF_SETTINGS["status_collection"] = f"test_status_{target_date}"

    def process_one(self, doc: dict) -> Optional[dict]:
        """处理一条文档，返回最终状态。graph.py集成入口。"""
        skip  = int(doc.get("from_batch_skip", 0))
        limit = int(doc.get("batch_limit", 30))
        logging.info(f"📄 处理文档: type={doc.get('type')}, skip={skip}, limit={limit}")
        initial_state: RAGState = {                          # 构造初始状态
            "current_doc": doc, "from_batch_skip": skip, "batch_limit": limit,
            "think_summary": "", "think_embedding": [], "semantic_uids": [],
            "flow_uids": [], "all_context_docs": [], "graph_context": "", "final_judgment": ""}
        try:
            return rag_graph.invoke(initial_state)           # 调图处理一条
        except Exception as e:
            logging.error(f"❌ RAG 处理失败 (skip={skip}): {e}"); logging.error(traceback.format_exc())
            try:                                             # 标记失败避免重复取
                history_db, _ = _get_mongo()
                history_db[f"test_status_{self.target_date}"].update_one(
                    {"_id": doc["_id"]}, {"$set": {"rag_processed": True, "rag_error": str(e)}})
            except: pass
            return None

    def process_next(self) -> Optional[dict]:
        """从MongoDB取下一条未处理文档并处理"""
        history_db, _ = _get_mongo()
        doc = history_db[f"test_status_{self.target_date}"].find_one(
            {"from_batch_skip": {"$exists": True}, "rag_processed": {"$ne": True}},
            sort=[("from_batch_skip", 1)])                   # 按编号升序取最早的
        if doc is None:
            logging.info("✅ 所有文档已处理完毕。"); return None
        return self.process_one(doc)

    def run(self, max_docs: int = 0):
        """批量循环：不停调process_next()直到没有文档或达到上限"""
        print(f"\n{'='*60}")
        print(f"  RAG Pipeline Start (Target: {self.target_date})")
        print(f"  Embedding: {EMBED_MODEL}")
        print(f"  Neo4j: {NEO4J_URI}")
        if max_docs > 0: print(f"  Max docs: {max_docs}")
        print(f"{'='*60}\n")
        _check_connectivity()                                # 启动前检查
        processed, start_time = 0, time.time()
        try:
            while True:
                result = self.process_next()                 # 取一条处理一条
                if result is None:                           # None=没有更多 或 处理失败
                    history_db, _ = _get_mongo()
                    remaining = history_db[f"test_status_{self.target_date}"].count_documents(
                        {"from_batch_skip": {"$exists": True}, "rag_processed": {"$ne": True}})
                    if remaining == 0:                       # 真没有了
                        logging.info("🎉 全部处理完毕。"); break
                    else:                                    # 是失败，还有剩余
                        logging.info(f"⏭️  跳过失败文档，剩余 {remaining} 条")
                        time.sleep(2); continue
                processed += 1
                elapsed = time.time() - start_time
                logging.info(f"── 已处理 {processed} 条 (累计 {elapsed:.0f}s, 均 {elapsed/processed:.1f}s/条) ──")
                if max_docs > 0 and processed >= max_docs:   # 达到上限
                    logging.info(f"🛑 已达到 --max-docs {max_docs} 上限，停止。"); break
                time.sleep(0.5)
        except KeyboardInterrupt:                            # Ctrl+C
            logging.info("\n🛑 收到中断信号 (Ctrl+C)，安全退出…")
        if _neo4j_driver: _neo4j_driver.close()
        elapsed = time.time() - start_time
        print(f"\n{'='*60}\n  RAG Pipeline Finished\n  共处理 {processed} 条 | 耗时 {elapsed:.0f}s")
        if processed > 0: print(f"  平均 {elapsed/processed:.1f}s / 条")
        print(f"{'='*60}\n")

    def reset(self, reset_neo4j: bool = False):
        """终极重置：清空所有 RAG 相关的计算字段 + 删知识库 + 可选清空 Neo4j"""
        history_db, _ = _get_mongo()
        status_col = history_db[f"test_status_{self.target_date}"]
        
        # 1. 抹除流水账表中的所有 RAG 痕迹（注意：这里已经补上了 rag_score）
        r_status = status_col.update_many(
            {}, 
            {"$unset": {
                "rag_processed": "",
                "rag_judgment": "",
                "rag_score": "",            # 🌟 罪魁祸首已加入清剿名单
                "rag_semantic_uids": "",
                "rag_flow_uids": "",
                "rag_updated_at": "",
                "rag_error": "",
                "think_summary": ""         # 把之前 LLM 生成的摘要也洗掉，防止残留
            }}
        )
        logging.info(f"🔄 [MongoDB] 已重置 {r_status.modified_count} 条文档的 RAG 处理标记及分数。")
        
        # 2. 彻底核平知识库（弹药库）
        kb_col = history_db["rag_knowledge_base"]
        r_kb = kb_col.delete_many({})
        logging.info(f"🗑️  [MongoDB] 已彻底清空 rag_knowledge_base (共删除 {r_kb.deleted_count} 条向量特征)。")
        
        # 3. 彻底清空 Neo4j 图谱（如果有指令）
        if reset_neo4j:
            try:
                driver = _get_neo4j()
                with driver.session() as s: 
                    # MATCH (n) DETACH DELETE n 是 Neo4j 的终极删库跑路指令，删掉所有节点和关系
                    s.run("MATCH (n) DETACH DELETE n")
                logging.info("🗑️  [Neo4j] 已彻底清空 Neo4j 知识图谱的所有节点与边。")
            except Exception as e: 
                logging.error(f"❌ 清空 Neo4j 失败: {e}")
                
        if _neo4j_driver: 
            _neo4j_driver.close()
            
        logging.info("✅ 终极重置完成！系统已恢复到一张白纸的状态。")

# ─── 模块级实例：graph.py 直接 from rag import rag_processor 就能用 ───
rag_processor = RAGProcessor()

# ─────────────────────────── 命令行入口 ───────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GYF RAG 工作流 — 网络流量异常检测增强推理")
    parser.add_argument("--max-docs", type=int, default=0, help="最多处理几条（0=不限）")
    parser.add_argument("--reset", action="store_true", help="重置所有RAG处理状态")
    parser.add_argument("--reset-neo4j", action="store_true", help="重置时同时清Neo4j")
    parser.add_argument("--target-date", type=str, default=TARGET_DATE, help=f"数据集日期（默认{TARGET_DATE}）")
    args = parser.parse_args()
    processor = RAGProcessor(target_date=args.target_date)
    if args.reset: processor.reset(reset_neo4j=args.reset_neo4j)  # python rag.py --reset --reset-neo4j
    else:          processor.run(max_docs=args.max_docs)          # python rag.py --max-docs 10