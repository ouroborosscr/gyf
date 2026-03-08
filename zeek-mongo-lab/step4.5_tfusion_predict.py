import os
import sys
import numpy as np
import logging
from pymongo import MongoClient
import torch
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score



# --- 1. 配置与环境 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# 【关键】将 tfusion 源码目录加入路径
sys.path.append(os.path.join(current_dir, "tfusion")) 
try:
    from model.trafformer import TFusion
except ImportError:
    logging.error("无法导入 TFusion 模型，请确保 tfusion 源码已放在正确路径下。")
    sys.exit(1)

try:
    from utils import config
except ImportError:
    config = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 集合配置 ---
MONGO_URI = config.DATABASE["mongo"]["uri"] if config else "mongodb://localhost:27017/"
DB_NAME = config.DATABASE["mongo"]["db_name"] if config else "zeek_db"
CONN_COL = "2_28_conn"
LABEL_COL = "2_28_labels"
PAYLOAD_COL = "2_28_payload"
PREDICT_COL = "2_28_predictions" # 存放最终打标结果的新集合

SEGMENT_LEN = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. 辅助函数 ---

def build_host_profiles(db):
    logging.info("正在构建全局主机交互画像 (Host Profiles)...")
    host_stats = {}
    for doc in db[CONN_COL].find({}, {"id_orig_h":1, "id_resp_h":1, "orig_pkts":1, "resp_pkts":1, "orig_ip_bytes":1, "resp_ip_bytes":1}):
        src = doc.get("id_orig_h")
        dst = doc.get("id_resp_h")
        
        if src not in host_stats: host_stats[src] = {'sf':0, 'sp':0, 'sb':0, 'rf':0, 'rp':0, 'rb':0}
        if dst not in host_stats: host_stats[dst] = {'sf':0, 'sp':0, 'sb':0, 'rf':0, 'rp':0, 'rb':0}
            
        host_stats[src]['sf'] += 1
        host_stats[src]['sp'] += doc.get('orig_pkts', 0)
        host_stats[src]['sb'] += doc.get('orig_ip_bytes', 0)
        
        host_stats[dst]['rf'] += 1
        host_stats[dst]['rp'] += doc.get('resp_pkts', 0)
        host_stats[dst]['rb'] += doc.get('resp_ip_bytes', 0)
        
    return host_stats

def extract_features(uid, conn_doc, db, host_stats):
    # 1. 交互特征 - 12维
    src = conn_doc.get("id_orig_h", "")
    dst = conn_doc.get("id_resp_h", "")
    s_stat = host_stats.get(src, {'sf':0, 'sp':0, 'sb':0, 'rf':0, 'rp':0, 'rb':0})
    d_stat = host_stats.get(dst, {'sf':0, 'sp':0, 'sb':0, 'rf':0, 'rp':0, 'rb':0})
    interact_raw = [
        s_stat['sf'], s_stat['sp'], s_stat['sb'], s_stat['rf'], s_stat['rp'], s_stat['rb'],
        d_stat['sf'], d_stat['sp'], d_stat['sb'], d_stat['rf'], d_stat['rp'], d_stat['rb']
    ]
    interact_feat = np.log1p(interact_raw)

    # 2. 非序列特征 - 6维
    num_pkts = conn_doc.get('orig_pkts', 0) + conn_doc.get('resp_pkts', 0)
    duration = conn_doc.get('duration', 0.0)
    sum_bytes = conn_doc.get('orig_ip_bytes', 0) + conn_doc.get('resp_ip_bytes', 0)
    
    payloads = list(db[PAYLOAD_COL].find({"uid": uid}).sort("ts", 1).limit(SEGMENT_LEN))
    if payloads:
        lengths = [p.get('len', 0) for p in payloads]
        min_len, max_len = min(lengths), max(lengths)
        mean_len = sum(lengths) / len(lengths)
    else:
        min_len = max_len = 0
        mean_len = sum_bytes / num_pkts if num_pkts > 0 else 0
        
    nonseq_feat = np.log1p([num_pkts, duration, sum_bytes, min_len, max_len, mean_len])

    # 3. 序列特征 - 100 x 2
    seq_feat = np.zeros((SEGMENT_LEN, 2), dtype=int)
    proto_str = conn_doc.get('proto', 'tcp')
    proto_offset = 0 if proto_str == 'tcp' else (1500 if proto_str == 'udp' else 3000)
    first_ts = payloads[0]['ts'] if payloads else 0.0
    
    for j, p in enumerate(payloads):
        if j >= SEGMENT_LEN: break
        l_token = proto_offset + min(p.get('len', 0), 1499) 
        t_delta = p['ts'] - first_ts
        if j == 0: t_token = 0
        elif t_delta <= 0.01: t_token = min(2 * j + 1, 199)
        else: t_token = min(2 * j, 199)
        seq_feat[j] = [l_token, t_token]

    return seq_feat, nonseq_feat, interact_feat

def safe_tfusion_inference(tfusion_model, seq_list, nonseq_list, interact_list):
    if len(seq_list) == 0:
        return np.array([])
    is_single = (len(seq_list) == 1)
    if is_single:
        seq_list = [seq_list[0], seq_list[0]]
        nonseq_list = [nonseq_list[0], nonseq_list[0]]
        interact_list = [interact_list[0], interact_list[0]]
        
    seq_t = torch.tensor(np.array(seq_list), dtype=torch.float32).to(DEVICE)
    nonseq_t = torch.tensor(np.array(nonseq_list), dtype=torch.float32).to(DEVICE)
    interact_t = torch.tensor(np.array(interact_list), dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        fused_emb = tfusion_model((seq_t, nonseq_t, interact_t)).cpu().numpy()
        
    if is_single:
        return fused_emb[[0]]
    return fused_emb

# --- 3. 主流程 ---

def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    
    # 0. 强制建立关键查询索引 (极其重要)
    logging.info("正在检查并建立 MongoDB 索引以加速查询 (若已有会瞬间跳过)...")
    db[CONN_COL].create_index("uid")
    db[PAYLOAD_COL].create_index([("uid", 1), ("ts", 1)])
    logging.info("索引准备完毕！")

    logging.info("获取标签数据...")
    labeled_docs = list(db[LABEL_COL].find({}, {"uid": 1, "label": 1}))
    # 直接保留原始字符串标签，如 'Benign', 'DDoS', 'Bot' 等
    labeled_dict = {d['uid']: d['label'] for d in labeled_docs}
    labeled_uids = list(labeled_dict.keys())
    
    host_stats = build_host_profiles(db)
    
    logging.info(f"初始化 tFusion 模型 (Device: {DEVICE})...")
    tfusion_model = TFusion(hidden=100).to(DEVICE)
    tfusion_model.eval()
    
    logging.info(f"提取训练集多模态特征 (共 {len(labeled_uids)} 条，采用游标批量加速)...")
    train_features, train_labels = [], []
    batch_seq_tr, batch_nonseq_tr, batch_interact_tr = [], [], []
    
    # 优化：不使用 for uid in uids 逐个 find_one，而是直接拉取一次大游标
    start_time = time.time()
    train_cursor = db[CONN_COL].find({"uid": {"$in": labeled_uids}}, no_cursor_timeout=True)
    
    processed_train = 0
    for conn_doc in train_cursor:
        uid = conn_doc["uid"]
        seq, nonseq, interact = extract_features(uid, conn_doc, db, host_stats)
        
        batch_seq_tr.append(seq)
        batch_nonseq_tr.append(nonseq)
        batch_interact_tr.append(interact)
        train_labels.append(labeled_dict[uid])
        
        processed_train += 1
        
        if len(batch_seq_tr) >= 2000:
            embs = safe_tfusion_inference(tfusion_model, batch_seq_tr, batch_nonseq_tr, batch_interact_tr)
            train_features.extend(embs)
            batch_seq_tr, batch_nonseq_tr, batch_interact_tr = [], [], []
            elapsed = time.time() - start_time
            logging.info(f"  已处理: {processed_train}/{len(labeled_uids)} ({processed_train/elapsed:.0f} 条/秒) | 已提炼特征: {len(train_features)}")

    if batch_seq_tr:
        embs = safe_tfusion_inference(tfusion_model, batch_seq_tr, batch_nonseq_tr, batch_interact_tr)
        train_features.extend(embs)
        
    train_cursor.close()

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    
    logging.info(f"特征提取完毕！使用 {len(train_features)} 条样本训练 Random Forest...")
    detector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) # 开启多线程训练
    detector.fit(train_features, train_labels)

    # 使用训练过的全量数据直接进行验证（查看模型对训练集的拟合程度）
    train_preds = detector.predict(train_features)

    # 计算并打印评估指标
    acc = accuracy_score(train_labels, train_preds)
    logging.info(f"🌟 全量数据自验证准确率 (Training Accuracy): {acc * 100:.4f}%")

    # 打印详细的分类报告
    report = classification_report(train_labels, train_preds)
    logging.info("\n" + "="*40 + "\n全量数据自验证分类报告:\n" + report + "="*40)
    
    logging.info("开始预测剩余未打标的流量...")
    # 使用游标拉取剩余数据
    cursor = db[CONN_COL].find({"uid": {"$nin": labeled_uids}}, no_cursor_timeout=True)
    
    batch_size = 2000
    batch_uids, batch_seq, batch_nonseq, batch_interact = [], [], [], []
    predictions_to_insert = []
    processed_count = 0
    start_time = time.time()
    
    for conn_doc in cursor:
        uid = conn_doc["uid"]
        seq, nonseq, interact = extract_features(uid, conn_doc, db, host_stats)
        
        batch_uids.append(uid)
        batch_seq.append(seq)
        batch_nonseq.append(nonseq)
        batch_interact.append(interact)
        
        if len(batch_uids) >= batch_size:
            fused_emb = safe_tfusion_inference(tfusion_model, batch_seq, batch_nonseq, batch_interact)
            preds = detector.predict(fused_emb)
            
            for idx, p_label in enumerate(preds):
                predictions_to_insert.append({
                    "uid": batch_uids[idx],
                    "label": p_label,  # 直接存入多分类标签，例如 'DDoS'
                    "match_type": "tfusion_predict",
                    "csv_src": "tfusion_inference"
                })
            
            db[PREDICT_COL].insert_many(predictions_to_insert)
            processed_count += len(batch_uids)
            elapsed = time.time() - start_time
            logging.info(f"推理入库中... 已完成 {processed_count} 条 | 速度: {processed_count/elapsed:.0f} 条/秒")
            
            batch_uids, batch_seq, batch_nonseq, batch_interact, predictions_to_insert = [], [], [], [], []

    if batch_uids:
        fused_emb = safe_tfusion_inference(tfusion_model, batch_seq, batch_nonseq, batch_interact)
        preds = detector.predict(fused_emb)
        for idx, p_label in enumerate(preds):
            predictions_to_insert.append({
                "uid": batch_uids[idx],
                "label": "Attack" if p_label == 1 else "Benign",
                "match_type": "tfusion_predict",
                "csv_src": "tfusion_inference"
            })
        db[PREDICT_COL].insert_many(predictions_to_insert)
        processed_count += len(batch_uids)

    cursor.close()
    logging.info(f"=========================================")
    logging.info(f"任务完成！总计基于 TFusion 预测了 {processed_count} 条无标签流量。")
    logging.info(f"预测结果已存入集合: {PREDICT_COL}")

if __name__ == "__main__":
    main()