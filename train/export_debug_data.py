import os
from transformers import AutoTokenizer
# 导入你原来的构建逻辑
from train_dapo_3_bug import build_grpo_dataset, MODEL_PATH, TARGET_DATE

def export_data():
    print("🚀 准备抽取 Debug 专用测试数据...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "left"
    
    # 我们只抽取 8 条数据，刚好够跑几个 Batch
    dataset = build_grpo_dataset(
        tokenizer=tokenizer, 
        target_date=TARGET_DATE, 
        range_start=1, 
        range_end=3000, 
        num_samples=8 
    )
    
    output_dir = "./debug_dataset_cache"
    dataset.save_to_disk(output_dir)
    print(f"✅ 数据已成功序列化并保存到: {output_dir}")
    print("👉 你现在可以断开 MongoDB，使用 debug 脚本进行单机测试了！")

if __name__ == "__main__":
    export_data()