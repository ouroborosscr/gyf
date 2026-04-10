import os
import sys
import logging
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# --- 环境设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from utils import config
except ImportError:
    logging.error("无法导入 utils.config，请确保在项目根目录下运行此脚本")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [Evaluate] - %(message)s')

# ================= 配置区 =================
TARGET_DATE = "3_1"  # 与你在 Test 脚本中保持一致
EVAL_START = 0      # 🌟 新增：指定评估的起始流编号 (比如从第 83 条开始)
EVAL_LIMIT = 636     # 指定评估的结束流编号。设为 None 或 0 则评估到最后
# ==========================================

def main():
    mongo_uri = config.DATABASE["mongo"]["uri"]
    zeek_db_name = config.DATABASE["mongo"]["db_name"]
    history_db_name = config.GYF_SETTINGS.get("history_db_name", "gyf_history")

    client = MongoClient(mongo_uri)
    zeek_db = client[zeek_db_name]
    history_db = client[history_db_name]

    # --- 集合名称定义 ---
    conn_col_name = f"{TARGET_DATE}_conn"
    label_col_name = f"{TARGET_DATE}_labels"
    tfusion_pred_col_name = f"{TARGET_DATE}_predictions"   
    llm_pred_col_name = f"{TARGET_DATE}_llm_predictions"   
    
    status_col_name = f"test_status_{TARGET_DATE}"
    status_eval_col_name = f"test_status_evaluate_{TARGET_DATE}" 

    conn_col = zeek_db[conn_col_name]
    label_col = zeek_db[label_col_name]
    tfusion_pred_col = zeek_db[tfusion_pred_col_name]
    llm_pred_col = zeek_db[llm_pred_col_name]
    status_col = history_db[status_col_name]
    status_eval_col = history_db[status_eval_col_name]

    # ================= 1. 获取模型分析范围与预测结果 =================
    control_doc = status_col.find_one({"current_id": {"$exists": True}})
    if not control_doc:
        logging.error(f"未找到测试进度记录 {status_col_name}，请先运行 GYF_Test.py")
        return
    
    current_id = int(control_doc.get("current_id", 0))
    total_flows = int(control_doc.get("flow_count", 0))
    logging.info(f"LLM 智能体当前共已分析流数量: {current_id} / {total_flows}")

    # --- 🌟 核心修改：确定本次要评估的精准区间 ---
    start_idx = max(0, EVAL_START) if EVAL_START else 0
    end_idx = min(current_id, EVAL_LIMIT) if (EVAL_LIMIT and EVAL_LIMIT > 0) else current_id
    eval_count = max(0, end_idx - start_idx)

    if eval_count == 0:
        logging.warning(f"当前指定评估的流数量为 0 (起始: {start_idx}, 结束: {end_idx})，无法计算。")
        return

    logging.info(f"已开启精准区间限制，实际评估范围: 第 {start_idx} 条 到 第 {end_idx} 条 (共 {eval_count} 条流量)。")

    logging.info("正在提取评估范围内的 UID...")
    # 🌟 核心修改：MongoDB 查询加入 .skip(start_idx)
    analyzed_cursor = conn_col.find({}, {"uid": 1}).sort("ts", 1).skip(start_idx).limit(eval_count)
    analyzed_uids = [doc["uid"] for doc in analyzed_cursor if "uid" in doc]
    
    suspicious_uids = set()
    alerts = status_col.find({"type": "suspicious_alert"})
    for alert in alerts:
        flows = alert.get("suspicious_flows", [])
        for f in flows:
            # 只统计在我们本次评估范围(analyzed_uids)内的告警流
            if "uid" in f and f["uid"] in analyzed_uids:
                suspicious_uids.add(f["uid"])
                
    logging.info(f"提取完毕，大模型在评估范围内共判定 {len(suspicious_uids)} 条流量为可疑 (Suspicious)。")

    # ================= 2. 联合获取 Ground Truth 真实标签 =================
    logging.info(f"正在从 {label_col_name} 和 {tfusion_pred_col_name} 联合提取真实标签...")
    true_labels = {}
    
    labels_cursor = label_col.find({"uid": {"$in": analyzed_uids}}, {"uid": 1, "label": 1})
    for doc in labels_cursor:
        true_labels[doc["uid"]] = doc["label"]
        
    logging.info(f" - 从 {label_col_name} 获取了 {len(true_labels)} 条 CSV 真实标签。")
    
    missing_uids = [uid for uid in analyzed_uids if uid not in true_labels]
    if missing_uids:
        tfusion_cursor = tfusion_pred_col.find({"uid": {"$in": missing_uids}}, {"uid": 1, "label": 1})
        count_tfusion = 0
        for doc in tfusion_cursor:
            true_labels[doc["uid"]] = doc["label"]
            count_tfusion += 1
        logging.info(f" - 从 {tfusion_pred_col_name} 补充获取了 {count_tfusion} 条 tFusion 预测标签。")

    logging.info(f"总计成功对齐到 {len(true_labels)} 条 Ground Truth。")

    # ================= 3. 数据集对齐与清洗 =================
    data = []
    for uid in analyzed_uids:
        raw_label = true_labels.get(uid, "Unknown")
        is_suspicious = (uid in suspicious_uids)
        data.append({
            "uid": uid,
            "true_label": raw_label,
            "pred_suspicious": is_suspicious
        })
    
    df = pd.DataFrame(data)

    df_eval = df[df["true_label"] != "Unknown"].copy()
    unlabeled_count = len(df) - len(df_eval)
    if unlabeled_count > 0:
        logging.info(f"已剔除 {unlabeled_count} 条既不在 labels 也不在 predictions 中的无标背景流量。")
    
    if len(df_eval) == 0:
        logging.warning("参与评估的流数量为 0，没有成功打标的流量。")
        return

    # 二值化处理
    df_eval["y_true"] = df_eval["true_label"].apply(lambda x: 0 if str(x).lower() == "benign" else 1)
    df_eval["y_pred"] = df_eval["pred_suspicious"].apply(lambda x: 1 if x else 0)

    # ================= 4. 计算指标与生成报告 =================
    y_true = df_eval["y_true"]
    y_pred = df_eval["y_pred"]

    actual_pos = sum(y_true == 1)
    actual_neg = sum(y_true == 0)
    pred_pos = sum(y_pred == 1)
    pred_neg = sum(y_pred == 0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0,0,0,0)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\n" + "="*55)
    print("               LLM 智能体流量分析评估报告")
    print("="*55)
    print(f" 评估范围: [{start_idx} ~ {end_idx}) 共 {eval_count} 条 (有效样本: {len(df_eval)} 条)")
    print("-" * 55)
    print(f"【真实情况】 正常 (Benign): {actual_neg} | 攻击 (Attack): {actual_pos}")
    if actual_pos > 0:
        print(f"  > 包含的攻击类型:\n{df_eval[df_eval['y_true']==1]['true_label'].value_counts().to_string()}")
    print(f"【模型预测】 正常 (Normal) : {pred_neg} | 可疑 (Alert) : {pred_pos}")
    print("-" * 55)
    print("【混淆矩阵 (Confusion Matrix)】")
    print(f"  TN (真阴性 - 正确识别为正常) : {tn}")
    print(f"  FP (假阳性 - 正常误报为可疑) : {fp}")
    print(f"  FN (假阴性 - 漏报了真实攻击) : {fn}")
    print(f"  TP (真阳性 - 成功揪出攻击流) : {tp}")
    print("-" * 55)
    print("【核心评估指标】")
    print(f"  - 准确率 (Accuracy) : {acc:.4f}")
    print(f"  - 精确率 (Precision): {prec:.4f}")
    print(f"  - 召回率 (Recall)   : {rec:.4f}")
    print(f"  - F1 分数 (F1-Score): {f1:.4f}")
    print("="*55 + "\n")

    # ================= 5. 保存 LLM 独立评估结果 =================
    llm_pred_col.delete_many({}) 
    records_to_insert = df_eval[["uid", "true_label", "pred_suspicious"]].to_dict(orient="records")
    if records_to_insert:
        llm_pred_col.insert_many(records_to_insert)
        logging.info(f"✅ 已将 {len(records_to_insert)} 条 LLM 二分类判定结果归档至: {llm_pred_col_name}")

    # ================= 6. 复制状态表并写入真实绝对区间 =================
    logging.info(f"正在生成包含真实评估区间的新集合: {status_eval_col_name}")
    status_eval_col.delete_many({}) 
    
    all_status_docs = list(status_col.find({}))
    eval_docs_to_insert = []
    
    for doc in all_status_docs:
        skip_val = doc.get("from_batch_skip", 0)
        
        # 🌟 核心修改：过滤掉不在评估范围内的批次
        if skip_val < start_idx or skip_val >= end_idx:
            continue

        new_doc = doc.copy()
        
        if new_doc.get("type") in ["suspicious_alert", "normal_log"]:
            limit_val = new_doc.get("batch_limit", 30)
            
            # 🌟 核心修改：切片偏移量修正！
            # 因为 analyzed_uids 是从 start_idx 开始抓的，所以需要把 skip_val 减去 start_idx 来对齐索引
            slice_start = skip_val - start_idx
            batch_uids = analyzed_uids[slice_start : slice_start + limit_val] 
            
            true_start = None
            true_end = None
            in_attack_block = False
            
            for i, uid in enumerate(batch_uids):
                raw_label = true_labels.get(uid, "Unknown")
                is_attack = (raw_label != "Unknown" and str(raw_label).lower() != "benign")
                
                # 绝对总编号 = 当前批次的跳过数(skip_val) + 局部索引(i) + 1
                absolute_index = skip_val + i + 1 
                
                if is_attack:
                    if not in_attack_block:
                        true_start = absolute_index
                        true_end = absolute_index
                        in_attack_block = True
                    else:
                        true_end = absolute_index
                else:
                    if in_attack_block:
                        break
                        
            new_doc["truly_suspicious_flows_start"] = true_start
            new_doc["truly_suspicious_flows_end"] = true_end
            
        eval_docs_to_insert.append(new_doc)
        
    if eval_docs_to_insert:
        status_eval_col.insert_many(eval_docs_to_insert)
        logging.info(f"✅ 已成功复制并标注 {len(eval_docs_to_insert)} 条评估区间内的状态记录到集合: {status_eval_col_name}")

if __name__ == "__main__":
    main()