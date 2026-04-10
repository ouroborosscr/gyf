import logging
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ================= 配置区 =================
TARGET_DATE = "3_1"
THRESHOLD = 0.9  # 🌟 RAG 置信度报警阈值
MONGO_URI = "mongodb://admin:gyf424201@localhost:62015"
# ==========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [RAG Eval] - %(message)s')

def main():
    client = MongoClient(MONGO_URI)
    zeek_db = client["zeek_data"]  # 请替换为你的实际数据库名 config.DATABASE["mongo"]["db_name"]
    history_db = client["gyf_history"]

    conn_col = zeek_db[f"{TARGET_DATE}_conn"]
    label_col = zeek_db[f"{TARGET_DATE}_labels"]
    tfusion_col = zeek_db[f"{TARGET_DATE}_predictions"]
    status_col = history_db[f"test_status_{TARGET_DATE}"]

    # 1. 提取所有跑过 RAG 且有分数的批次
    rag_docs = list(status_col.find({"rag_processed": True, "rag_score": {"$ne": -1.0}}).sort("from_batch_skip", 1))
    if not rag_docs:
        logging.error("没有找到 RAG 处理记录！请确认 rag.py 是否成功写入 rag_score。")
        return

    logging.info(f"提取到 {len(rag_docs)} 个 RAG 研判批次，正在对齐底层真实标签...")

    # 2. 预先加载所有的真实标签 (为了速度，把相关的 UID 标签全拉到内存字典里)
    # 获取最大覆盖范围
    max_idx = max([doc.get("from_batch_skip", 0) + doc.get("batch_limit", 30) for doc in rag_docs])
    analyzed_cursor = conn_col.find({}, {"uid": 1}).sort("ts", 1).limit(max_idx)
    uid_list = [d["uid"] for d in analyzed_cursor if "uid" in d]

    true_labels = {}
    for l_doc in label_col.find({"uid": {"$in": uid_list}}, {"uid": 1, "label": 1}):
        true_labels[l_doc["uid"]] = l_doc["label"]
    for t_doc in tfusion_col.find({"uid": {"$in": uid_list}}, {"uid": 1, "label": 1}):
        if t_doc["uid"] not in true_labels:
            true_labels[t_doc["uid"]] = t_doc["label"]

    # 3. 核心比对逻辑 (Batch-Level)
    y_true_batch = []
    y_pred_batch = []
    bad_cases_fn = []
    bad_cases_fp = []

    for doc in rag_docs:
        skip = doc.get("from_batch_skip", 0)
        limit = doc.get("batch_limit", 30)
        score = doc.get("rag_score", 0.0)

        # 提取这个批次包含的所有 UID
        batch_uids = uid_list[skip : skip + limit]
        
        # 判断这个批次底层是否真的含有攻击流
        batch_has_attack = False
        attack_types_in_batch = set()
        for uid in batch_uids:
            label = str(true_labels.get(uid, "Unknown")).lower()
            if label != "unknown" and label != "benign":
                batch_has_attack = True
                attack_types_in_batch.add(label)

        # 真值 (Ground Truth) 和 预测值 (Prediction)
        gt = 1 if batch_has_attack else 0
        pred = 1 if score >= THRESHOLD else 0

        y_true_batch.append(gt)
        y_pred_batch.append(pred)

        # 记录 Bad Cases
        if gt == 1 and pred == 0:
            bad_cases_fn.append((skip, score, list(attack_types_in_batch)))
        elif gt == 0 and pred == 1:
            bad_cases_fp.append((skip, score))

    # 4. 生成报告
    cm = confusion_matrix(y_true_batch, y_pred_batch, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)

    print("\n" + "="*60)
    print(f"      🛡️  RAG 增强架构评估报告 (批次级别)")
    print("="*60)
    print(f" 评估阈值 (Threshold): {THRESHOLD}")
    print(f" 评估批次总数: {len(y_true_batch)}")
    print("-" * 60)
    print(f" 【真实环境】 纯正常批次: {sum(1 for y in y_true_batch if y==0)} | 包含攻击的批次: {sum(1 for y in y_true_batch if y==1)}")
    print(f" 【RAG 报警】 放行批次: {sum(1 for y in y_pred_batch if y==0)}   | 拦截批次: {sum(1 for y in y_pred_batch if y==1)}")
    print("-" * 60)
    print(" 【混淆矩阵】")
    print(f"   TN (防守成功，未打扰) : {tn}")
    print(f"   FP (狼来了，过度敏感) : {fp}")
    print(f"   FN (防线被破，致命漏报) : {fn}")
    print(f"   TP (火眼金睛，成功抓获) : {tp}")
    print("-" * 60)
    print(f"   - 准确率 (Accuracy) : {accuracy_score(y_true_batch, y_pred_batch):.4f}")
    print(f"   - 精确率 (Precision): {precision_score(y_true_batch, y_pred_batch, zero_division=0):.4f}")
    print(f"   - 召回率 (Recall)   : {recall_score(y_true_batch, y_pred_batch, zero_division=0):.4f}")
    print(f"   - F1 分数 (F1-Score): {f1_score(y_true_batch, y_pred_batch, zero_division=0):.4f}")
    print("="*60)

    if bad_cases_fn:
        print("\n🚨 [致命漏报 FN Top 5] (批次内含攻击，但 RAG 给了低分):")
        for skip, score, types in bad_cases_fn[:5]:
            print(f"   - 批次 Skip: {skip} | RAG 得分: {score} | 实际包含的攻击: {types}")
            
    if bad_cases_fp:
        print("\n⚠️ [烦人误报 FP Top 5] (全是正常流量，但 RAG 瞎报警):")
        for skip, score in bad_cases_fp[:5]:
            print(f"   - 批次 Skip: {skip} | RAG 得分: {score}")

if __name__ == "__main__":
    main()