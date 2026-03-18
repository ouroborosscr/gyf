import os
import sys
import argparse         
import torch            
import requests         
from tqdm import tqdm   # 🚀 加回进度条


# ==========================================
# 🐒 神级猴子补丁
# ==========================================
import transformers.configuration_utils
if not hasattr(transformers.configuration_utils, "ALLOWED_LAYER_TYPES"):
    transformers.configuration_utils.ALLOWED_LAYER_TYPES = (object,)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

import re               
import json             
import random   
import numpy as np        
import logging          
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  
from trl import GRPOConfig, GRPOTrainer  
from datasets import Dataset             
from pymongo import MongoClient          
from peft import LoraConfig
import gc

# 自定义显存清理回调（升级版）
class MemoryCleanupCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # 每次梯度更新后清理一次
        gc.collect()
        torch.cuda.empty_cache()
        
    def on_epoch_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("🧹 Epoch 结束，已强制清理显存碎片及垃圾回收。")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)               
if project_root not in sys.path:
    sys.path.append(project_root)  

try:
    from utils import config  
except ImportError:
    print("无法导入 utils.config，请确保路径正确")
    sys.exit(1)  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [GRPO] - %(message)s')

# ==========================================
# 【环境自适应感知】
# ==========================================
is_distributed = "LOCAL_RANK" in os.environ
visible_gpus = torch.cuda.device_count()

if is_distributed:
    local_rank = int(os.environ["LOCAL_RANK"])
    safe_device_id = local_rank % visible_gpus if visible_gpus > 0 else 0
    current_device = f"cuda:{safe_device_id}"
    model_device_map = {"": safe_device_id}
    torch.cuda.set_device(safe_device_id)
    torch.cuda.get_device_properties(safe_device_id)
    torch.tensor([0.0]).to(current_device)
    logging.info(f"🌐 检测到分布式环境，当前进程 {local_rank} 已绑定至 {current_device}")
else:
    current_device = "cuda:0" if visible_gpus > 0 else "cpu"
    model_device_map = "auto"
    logging.info(f"🖥️ 未检测到分布式环境，采用自适应加载，绑定至 {current_device}")
# else:
#     current_device = "cuda:0" if visible_gpus > 0 else "cpu"
    
#     # 终极精准切片（完美适配 Qwen-9B 架构）
#     model_device_map = {
#         "model.embed_tokens": 0,  # 词表嵌入层：绑在卡 0
#         "model.rotary_emb": 0,    # 🚨 修复关键：旋转位置编码也必须绑在卡 0！
#         "model.norm": 1,          # 尾部的 LayerNorm 放卡 1
#         "lm_head": 1              # 最后的输出头放卡 1
#     }
    
#     # Qwen-9B 一共有 32 层隐藏层，完美对半劈开 (0-15去卡0，16-31去卡1)
#     for i in range(32):
#         if i < 8:
#             model_device_map[f"model.layers.{i}"] = 0
#         else:
#             model_device_map[f"model.layers.{i}"] = 1
            
#     logging.info(f"🖥️ 未检测到分布式环境，采用【精准强制切片】，底层与位置编码强制绑定至 cuda:0")

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"  
AGENT_RRM_PATH = "/date/sunchengrui/models/Agent-RRM"                               
OUTPUT_DIR = "./qwen-agent-grpo-output"             
TARGET_DATE = "3_1"  

RRM_API_URL = "http://localhost:8123/v1/completions"

ANALYSIS_PROMPT = """我在进行流量检测,现在我将给你数条json格式的流量数据,这些流量数据中可能存在多个攻击,也可能不存在攻击。
攻击的类型为”内网穿透“”渗透隐蔽信道“”加密通信“。
我希望你能找到最靠前的攻击流量的编号和这个攻击包含连续的几条流量,并返回其攻击类型。

【工具调用严格约束】：
如果你决定调用 `report_suspicious_traffic_tool` 报警，请务必确保传入的 JSON 参数键名拼写绝对准确！
必须严格使用 `is_suspicious` (布尔值)、`suspicious_flows_start` (整数) 和 `suspicious_flows_end` (整数)。绝对不能拼错或加入多余空格（例如绝不能写成 is_s suspicious），否则会导致系统崩溃！

以下是本次需要分析的流量数据：
{flow_data_json}"""

def ensure_indexes(db, target_date):
    conn_col_name = f"{target_date}_conn"
    payload_col_name = f"{target_date}_payload"
    label_col_name = f"{target_date}_labels"
    pred_col_name = f"{target_date}_predictions"

    conn_col = db[conn_col_name]
    if "ts_1" not in conn_col.index_information(): conn_col.create_index([("ts", 1)])
    payload_col = db[payload_col_name]
    if "uid_1" not in payload_col.index_information():
        payload_col.create_index([("uid", 1)])
        payload_col.create_index([("ts", 1)])
    label_col = db[label_col_name]
    if "uid_1" not in label_col.index_information(): label_col.create_index([("uid", 1)])
    pred_col = db[pred_col_name]
    if "uid_1" not in pred_col.index_information(): pred_col.create_index([("uid", 1)])

def fetch_flows_for_dataset(db, target_date, skip_idx, limit_val, max_payload_len=2000):
    conn_col = db[f"{target_date}_conn"]        
    payload_col = db[f"{target_date}_payload"]  
    cursor = conn_col.find({}).sort("ts", 1).skip(skip_idx).limit(limit_val)
    export_data = []      
    processed_count = 0   
    for conn_doc in cursor:
        uid = conn_doc.get("uid")  
        if not uid: continue       
        payload_cursor = payload_col.find({"uid": uid}).sort("ts", 1)
        full_hex, full_decoded = [], []  
        packet_count = 0                 
        current_len = 0                  
        for p_doc in payload_cursor:
            packet_count += 1
            if max_payload_len > 0 and current_len >= max_payload_len: continue
            if "payload" in p_doc: full_hex.append(p_doc["payload"])
            if "payload_decoded" in p_doc:
                decoded_fragment = p_doc["payload_decoded"]
                full_decoded.append(decoded_fragment)
                current_len += len(decoded_fragment)  
        entry = conn_doc.copy()  
        if "_id" in entry:
            entry["mongo_id"] = str(entry["_id"])  
            del entry["_id"]                       
        if "ts_date" in entry and hasattr(entry["ts_date"], "isoformat"):
            entry["ts_date"] = entry["ts_date"].isoformat()  
        entry["batch_index"] = processed_count + 1
        entry["packet_count_captured"] = packet_count  
        hex_str = "".join(full_hex)
        decoded_str = "".join(full_decoded)
        if max_payload_len > 0:
            if len(hex_str) > max_payload_len: hex_str = hex_str[:max_payload_len] + "...[TRUNCATED]"
            if len(decoded_str) > max_payload_len: decoded_str = decoded_str[:max_payload_len] + "...[TRUNCATED]"
        entry["stream_payload_hex"] = hex_str            
        entry["stream_payload_decoded"] = decoded_str    
        export_data.append(entry)                        
        processed_count += 1
    return export_data  

def build_grpo_dataset(target_date=TARGET_DATE, range_start=1, range_end=3000, num_samples=200):
    logging.info(f"📊 正在从 MongoDB 构建训练数据集 (随机抽取 {num_samples} 组)...")
    mongo_uri = config.DATABASE["mongo"]["uri"]  
    db_name = config.DATABASE["mongo"]["db_name"]
    client = MongoClient(mongo_uri)              
    db = client[db_name]
    label_col = db[f"{target_date}_labels"]
    pred_col = db[f"{target_date}_predictions"]
    ensure_indexes(db, target_date)  
    
    prompts, ground_truths = [], []
    for _ in tqdm(range(num_samples), desc="正在抽取流量数据"):
        L = 25
        min_skip = max(0, range_start - 1)
        max_skip = max(min_skip, range_end - L)
        S = random.randint(min_skip, max_skip)
        
        flows = fetch_flows_for_dataset(db, target_date, S, L)
        if not flows: continue  
            
        flows_json = json.dumps(flows, ensure_ascii=False, indent=2)
        final_prompt = ANALYSIS_PROMPT.replace("{flow_data_json}", flows_json)
        prompts.append(final_prompt)
        
        uids = [f["uid"] for f in flows]  
        true_labels = {}                  
        for doc in label_col.find({"uid": {"$in": uids}}, {"uid": 1, "label": 1}): true_labels[doc["uid"]] = doc["label"]
        missing_uids = [u for u in uids if u not in true_labels]
        if missing_uids:
            for doc in pred_col.find({"uid": {"$in": missing_uids}}, {"uid": 1, "label": 1}): true_labels[doc["uid"]] = doc["label"]
        
        flow_labels = []
        true_start, true_end = None, None
        in_first_attack_block, first_attack_ended = False, False
        
        for i, uid in enumerate(uids):
            lbl = true_labels.get(uid, "Unknown")
            is_attack = (lbl != "Unknown" and str(lbl).lower() != "benign")
            flow_labels.append(1 if is_attack else 0)
            local_idx = i + 1  
            if is_attack and not first_attack_ended:
                if not in_first_attack_block:
                    true_start = true_end = local_idx
                    in_first_attack_block = True
                else:
                    true_end = local_idx
            elif not is_attack and in_first_attack_block:
                first_attack_ended = True 
                    
        ground_truths.append({
            "start": true_start, "end": true_end, "skip_val": S, "limit_val": L, "flow_labels": flow_labels      
        })
        
    logging.info(f"✅ 数据集构建完成！成功生成 {len(prompts)} 条训练样本。")
    return Dataset.from_dict({"prompt": prompts, "ground_truth": ground_truths})

import math
import datetime

def format_reward_func(completions, **kwargs):
    rank = os.environ.get("LOCAL_RANK", "0")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[🕒 {current_time}] 🚀 汇报：Rank {rank} 已经完成本轮 Generate，正在请求 API 裁判打分！", flush=True)
    rewards = []
    for comp in completions:
        if re.search(r"<think>.*?</think>", comp, re.DOTALL): rewards.append(1.0)  
        else: rewards.append(-1.0) 
    return rewards

def correctness_reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        true_start, true_end = truth.get("start"), truth.get("end")
        flow_labels = truth.get("flow_labels", []) 
        start_match = re.search(r'(?:start|suspicious_flows_start)[\"\']?\s*[:=]\s*(\d+)', comp)
        end_match = re.search(r'(?:end|suspicious_flows_end)[\"\']?\s*[:=]\s*(\d+)', comp)

        if not (start_match and end_match):
            rewards.append(1.0 if true_start is None else -1.0); continue
        pred_start, pred_end = int(start_match.group(1)), int(end_match.group(1))
        if pred_start > pred_end or pred_start < 1:
            rewards.append(-1.0); continue
        if true_start is None:
            rewards.append(-1.0); continue
        if pred_start == true_start and pred_end == true_end:
            rewards.append(1.0); continue

        if pred_end < true_start or pred_start > true_end:
            if pred_end < true_start:
                dist = true_start - pred_end  
                score = -0.5 + 0.4 * math.exp(-0.2 * dist)
            else:
                dist = pred_start - true_end
                score = -0.9 + 0.4 * math.exp(-0.2 * dist)
            rewards.append(score); continue

        if pred_start <= true_start and pred_end >= true_end:
            penalty = 0.0  
            for i in range(pred_start, pred_end + 1):
                if i >= true_start and i <= true_end: continue 
                idx = i - 1  
                if 0 <= idx < len(flow_labels) and flow_labels[idx] == 1: penalty += 0.2 
                else: penalty += 1.0 
            score = 0.5 + 0.4 * math.exp(-0.2 * penalty)
            rewards.append(score); continue

        if pred_start >= true_start and pred_end <= true_end:
            miss_front = pred_start - true_start 
            miss_back = true_end - pred_end      
            penalty = miss_front * 1.0 + miss_back * 0.2
            score = 0.1 + 0.4 * math.exp(-0.3 * penalty)
            rewards.append(score); continue

        intersection = min(pred_end, true_end) - max(pred_start, true_start) + 1  
        union = max(pred_end, true_end) - min(pred_start, true_start) + 1         
        iou = intersection / union if union > 0 else 0                            
        score = -0.1 + 0.2 * iou
        rewards.append(score)
    return rewards

def logic_reward_func(completions, **kwargs):
    rewards = []
    for comp in completions:
        rm_prompt = f"Please evaluate the following agent reasoning and action:\n{comp}\n\n"
        try:
            payload = {
                "model": AGENT_RRM_PATH,
                "prompt": rm_prompt,
                "max_tokens": 150,
                "temperature": 0.0, 
            }
            response = requests.post(RRM_API_URL, json=payload, timeout=60)
            response.raise_for_status() 
            
            rm_output = response.json()["choices"][0]["text"]
            score_match = re.search(r'<score>\s*([0-9.]+)\s*</score>', rm_output)
            if score_match: rewards.append(float(score_match.group(1)))
            else: rewards.append(0.0) 
        except Exception as e:
            rewards.append(0.0)     
    return rewards

def main():
    parser = argparse.ArgumentParser(description="Qwen GRPO 训练脚本")
    parser.add_argument("--use_4bit", action="store_true", help="是否启用 4-bit QLoRA")
    parser.add_argument("--use_lora", action="store_true", help="是否启用 LoRA")
    parser.add_argument("--use_vllm", action="store_true", help="是否启用 vLLM")
    parser.add_argument("--use_flash_attn", action="store_true", help="是否启用 FA2")
    
    # 🚀 【新增 1】：断点重连参数
    # 不传就不重连；传 `--resume` 就自动找最新断点；传 `--resume ./xxx` 就加载特定断点
    parser.add_argument("--resume", type=str, nargs='?', const='True', default=None, help="从断点恢复训练")
    
    args = parser.parse_args()

    train_success = False
    try:
        if is_distributed:
            import torch.distributed as dist
            from datetime import datetime
            print(f"\n[🕒 {datetime.now().strftime('%H:%M:%S')}] [Rank {local_rank}] 🔌 准备开始 NCCL 底层握手...", flush=True)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            dummy_tensor = torch.tensor([1.0]).to(current_device)
            dist.all_reduce(dummy_tensor)
            print(f"[🕒 {datetime.now().strftime('%H:%M:%S')}] [Rank {local_rank}] 🎉 跨卡握手成功！")

        if args.use_4bit and not args.use_lora:
            sys.exit(1)

        logging.info("正在加载 Qwen3.5-9B 策略模型 ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if args.use_flash_attn:
            tokenizer.padding_side = "right" 
        else:
            tokenizer.padding_side = "left"

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": model_device_map,  
            "low_cpu_mem_usage": True,
        }

        if args.use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if args.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["quantization_config"] = bnb_config
        
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

        # 🚀 【关键修复：为 QLoRA 穿上防弹衣】
        if args.use_4bit:
            from peft import prepare_model_for_kbit_training
            # 这一步会自动将 LayerNorm 层转换为 fp32，防止训练几步后出现 NaN 和 Inf 崩溃
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        if args.use_lora:
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type="CAUSAL_LM",
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        else:
            peft_config = None 
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        dataset = build_grpo_dataset(target_date=TARGET_DATE, range_start=1, range_end=3000, num_samples=200)

        vllm_config = {}
        if args.use_vllm:
            logging.info(f"🚀 状态：检测到 --use_vllm，已唤醒 vLLM 引擎（与 PyTorch 共享显卡 {current_device}）！")
            vllm_config = {
                "use_vllm": True,
                "vllm_gpu_memory_utilization": 0.5, 
            }
        else:
            vllm_config = {"use_vllm": False}

        training_args = GRPOConfig(
            output_dir=OUTPUT_DIR,            
            learning_rate=5e-6,               
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",       
            logging_steps=1, 
            
            # 🚀 【修改】：既然长期训练，总步数必须调大
            max_steps=500,                     
            per_device_train_batch_size=1,    
            gradient_accumulation_steps=4,    

            
            num_generations=4,      
            max_completion_length=512,  
            bf16=True,                        
            gradient_checkpointing=True,      
            report_to="tensorboard",
            logging_dir="./runs/qwen-grpo-logs", # 告诉它曲线数据存在哪               
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            # repetition_penalty=1.05,
            repetition_penalty=1,
            optim="paged_adamw_8bit",
            
            # ==========================================
            # 🚀 【新增 2】：断点自动保存策略
            # ==========================================
            save_strategy="steps",
            save_steps=1,             # 每 10 步在 OUTPUT_DIR 保存一个 checkpoint
            save_total_limit=3,        # 硬盘保护：最多只保留最近的 3 个断点

            deepspeed="ds_config.json",
            
            **vllm_config
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward_func, correctness_reward_func, logic_reward_func],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,  
            callbacks=[MemoryCleanupCallback()]  # 🚀 【关键修复】：把你的垃圾回收器挂载上去！
        )

        # ==========================================
        # 🚀 【新增 3】：解析断点并注入训练引擎
        # ==========================================
        resume_checkpoint = None
        if args.resume is not None:
            if args.resume == 'True':
                resume_checkpoint = True
                logging.info(f"🔄 自动模式：正尝试从 {OUTPUT_DIR} 寻找最新的 Checkpoint...")
            else:
                resume_checkpoint = args.resume
                logging.info(f"🔄 手动模式：正从指定路径恢复: {resume_checkpoint}")

        logging.info("🚀 开始 GRPO 强化学习训练...")
        
        # 将重连参数传给 train 方法
        trainer.train(resume_from_checkpoint=resume_checkpoint)  
        
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_agent_model")) 
        train_success = True  
        
    except Exception as e:
        logging.error(f"❌ 训练过程中发生异常崩溃: {e}")
        raise e
        
    finally:
        if is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                if train_success:
                    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# 全新开始训练
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_grpo.py --use_lora --use_4bit
# 场景 2：跑一半断电了，或者爆显存崩溃了
# 直接在后面加上 --resume。代码会自动去 ./qwen-agent-grpo-output 目录里找到步数最大的那个断点，无缝接着跑，连进度条的数字都能完美续上：
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_grpo.py --use_lora --use_4bit --resume
# 场景 3：你想回滚到某一个特定的历史步数（比如第 20 步）
# 指定具体的断点文件夹路径即可：
# CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 train_grpo.py --use_lora --use_4bit --resume ./qwen-agent-grpo-output/checkpoint-20
