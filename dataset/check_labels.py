from pymongo import MongoClient
import pprint

# --- 配置 ---
MONGO_URI = "mongodb://admin:gyf424201@localhost:62015/"
DB_NAME = "zeek_analysis"
COLLECTION = "conn_test_auto"  # 你的集合名
# -----------

def check_db_status():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION]
    
    # 1. 统计总数 vs 已打标数
    total = col.count_documents({})
    labeled = col.count_documents({"label": {"$exists": True}})
    
    print("=" * 40)
    print(f"📊 数据库状态报告: {COLLECTION}")
    print("=" * 40)
    print(f"总记录数: {total}")
    print(f"已打标数: {labeled}")
    print(f"打标覆盖率: {labeled/total*100:.2f}%")
    
    if labeled == 0:
        print("\n❌ 警告：数据库里没有发现任何 'label' 字段！")
        print("请确认你是否完整运行了 apply_labels_final.py 并且没有报错。")
        return

    # 2. 看看都有哪些标签 (分类统计)
    print("\n📈 标签分布统计:")
    pipeline = [
        {"$match": {"label": {"$exists": True}}},
        {"$group": {"_id": "$label", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    results = list(col.aggregate(pipeline))
    
    for item in results:
        print(f"  - {item['_id']}: {item['count']}")

    # 3. 抽样查看一条完整数据
    print("\n🔍 样本数据 (1条):")
    sample = col.find_one({"label": {"$exists": True}})
    # 只打印关键字段
    pprint.pprint({
        "ts": sample.get('ts'),
        "id": sample.get('id', sample.get('id.resp_p')), # 兼容显示
        "proto": sample.get('proto'),
        "LABEL": sample.get('label'),       # <--- 这就是我们要的
        "META": sample.get('label_meta')    # <--- 这是匹配详情
    })

if __name__ == "__main__":
    check_db_status()