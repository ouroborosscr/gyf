import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import sys
import logging
import time

# --- 环境设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from utils import config
except ImportError:
    logging.error("无法导入 utils.config，脚本退出")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 核心配置
# ==========================================
TIME_OFFSET_HOURS = 12 
SEARCH_WINDOW = 1 

# 开启碎片回收模式
ENABLE_FRAGMENT_RECOVERY = True 

# 集合名称
CONN_COL_NAME = "3_1_conn"
LABEL_COL_NAME = "3_1_labels"

def parse_cic_timestamp(ts_str):
    try:
        dt = datetime.strptime(ts_str, "%d/%m/%Y %H:%M:%S")
        return dt + timedelta(hours=TIME_OFFSET_HOURS)
    except ValueError:
        return None

def get_best_match_score(candidate_doc, csv_row):
    """ 打分逻辑保持不变 """
    c_fwd = csv_row.get('Tot Fwd Pkts', 0)
    c_bwd = csv_row.get('Tot Bwd Pkts', 0)
    c_dur = csv_row.get('Flow Duration', 0) / 1_000_000.0
    c_proto_num = csv_row.get('Protocol', 6)
    
    z_orig = candidate_doc.get('orig_pkts', 0)
    z_resp = candidate_doc.get('resp_pkts', 0)
    z_dur = candidate_doc.get('duration', 0.0) or 0.0
    z_proto = candidate_doc.get('proto', 'tcp')

    proto_map = {6: 'tcp', 17: 'udp', 1: 'icmp'}
    expected_proto = proto_map.get(c_proto_num, 'tcp')
    
    proto_penalty = 0
    if z_proto != expected_proto:
        proto_penalty = 1000 

    pkt_diff = abs(z_orig - c_fwd) + abs(z_resp - c_bwd)
    dur_diff = abs(z_dur - c_dur)
    
    return proto_penalty + (pkt_diff * 10) + dur_diff

def main():
    csv_filename = "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"
    if os.path.exists(csv_filename):
        csv_path = os.path.abspath(csv_filename)
    elif os.path.exists(os.path.join("pcap", csv_filename)):
        csv_path = os.path.abspath(os.path.join("pcap", csv_filename))
    else:
        possible_path = "/date/sunchengrui/gyf/zeek-mongo-lab/" + csv_filename
        if os.path.exists(possible_path):
            csv_path = possible_path
        else:
            logging.error(f"找不到 CSV 文件: {csv_filename}")
            return

    try:
        mongo_uri = config.DATABASE["mongo"]["uri"]
        db_name = config.DATABASE["mongo"]["db_name"]
        client = MongoClient(mongo_uri)
        db = client[db_name]
        conn_col = db[CONN_COL_NAME]
        label_col = db[LABEL_COL_NAME]
        
        # 确保索引支持碎片查询：我们需要 id_orig_h 和 id_resp_h
        # 之前的脚本已经保留了这些字段
        # 建议检查是否存在 (id_orig_h, id_resp_h, ts) 的索引，如果没有，查询会变慢
        # 但考虑到我们是在极小的时间窗口查询，现有的 id_resp_p 索引可能也够用了
        
    except Exception as e:
        logging.error(f"数据库连接失败: {e}")
        return

    chunk_size = 5000
    total_processed = 0
    total_matched = 0
    total_fragments = 0 # 统计回收了多少碎片
    batch_labels = []
    
    logging.info(f"开始打标 (开启碎片回收模式)...")
    
    start_time = time.time()
    csv_reader = pd.read_csv(csv_path, chunksize=chunk_size)

    for chunk in csv_reader:
        for _, row in chunk.iterrows():
            try:
                dt_obj = parse_cic_timestamp(row['Timestamp'])
                if not dt_obj: continue
                
                ts_target = dt_obj.timestamp()
                dst_port = int(row['Dst Port'])
                
                # 计算流的结束时间 (用于寻找碎片)
                flow_duration_sec = row['Flow Duration'] / 1_000_000.0
                ts_end = ts_target + flow_duration_sec
                
                # --- Step 1: 寻找最佳锚点 (Anchor) ---
                query = {
                    "id_resp_p": dst_port,
                    "ts": { 
                        "$gte": ts_target - SEARCH_WINDOW, 
                        "$lte": ts_target + SEARCH_WINDOW
                    }
                }
                
                # 获取更多字段用于碎片匹配
                candidates = list(conn_col.find(query, {
                    "uid": 1, "ts": 1, "proto": 1,
                    "orig_pkts": 1, "resp_pkts": 1, "duration": 1,
                    "id_orig_h": 1, "id_resp_h": 1 # 必须取 IP
                }))
                
                if not candidates:
                    total_processed += 1
                    continue
                
                best_match = None
                best_score = float('inf')
                
                for doc in candidates:
                    score = get_best_match_score(doc, row)
                    if score < best_score:
                        best_score = score
                        best_match = doc
                
                if best_match:
                    # 添加锚点记录
                    batch_labels.append({
                        "uid": best_match["uid"],
                        "label": row['Label'],
                        "match_type": "anchor", # 标记为主体
                        "match_score": best_score,
                        "csv_src": "cicids2018"
                    })
                    total_matched += 1
                    
                    # --- Step 2: 顺藤摸瓜，回收碎片 ---
                    # 只有当 CSV 显示这是一个持续一段时间的流时才尝试回收
                    if ENABLE_FRAGMENT_RECOVERY and flow_duration_sec > 1.0:
                        anchor_src_ip = best_match.get("id_orig_h")
                        anchor_dst_ip = best_match.get("id_resp_h")
                        anchor_proto = best_match.get("proto")
                        
                        if anchor_src_ip and anchor_dst_ip:
                            # 查询条件：完全相同的五元组，且时间在流的生命周期内
                            # 注意：我们放宽一点时间窗口，防止边界误差
                            sibling_query = {
                                "id_orig_h": anchor_src_ip,
                                "id_resp_h": anchor_dst_ip,
                                "id_resp_p": dst_port,
                                "proto": anchor_proto,
                                "ts": {
                                    "$gte": ts_target - 1.0,  # 稍微前一点
                                    "$lte": ts_end + 5.0      # 结束时间后再宽限一点
                                },
                                "uid": {"$ne": best_match["uid"]} # 排除锚点自己
                            }
                            
                            siblings = list(conn_col.find(sibling_query, {"uid": 1}))
                            
                            for sib in siblings:
                                batch_labels.append({
                                    "uid": sib["uid"],
                                    "label": row['Label'],
                                    "match_type": "fragment", # 标记为碎片
                                    "anchor_uid": best_match["uid"], # 记录它属于哪个主体
                                    "csv_src": "cicids2018_expansion"
                                })
                                total_matched += 1
                                total_fragments += 1

                total_processed += 1

            except Exception:
                continue
        
        # 批量写入
        if batch_labels:
            try:
                label_col.insert_many(batch_labels, ordered=False)
                batch_labels = []
            except Exception:
                pass
        
        # 进度
        elapsed = time.time() - start_time
        if total_processed % 5000 == 0:
            speed = total_processed / elapsed if elapsed > 0 else 0
            logging.info(f"进度: {total_processed} | 总命中: {total_matched} (含碎片: {total_fragments}) | 速度: {speed:.0f}/s")

    logging.info("="*40)
    logging.info(f"任务完成 | CSV行数: {total_processed} | 数据库命中数: {total_matched} | 回收碎片数: {total_fragments}")

if __name__ == "__main__":
    main()