import os
import sys
import argparse         # 用于解析命令行参数
import torch            # PyTorch深度学习框架

# ==========================================
# 🐒 【新增】：神级猴子补丁，解决 vLLM 与最新 Transformers 的兼容性死锁！
# ==========================================
import transformers.configuration_utils
if not hasattr(transformers.configuration_utils, "ALLOWED_LAYER_TYPES"):
    transformers.configuration_utils.ALLOWED_LAYER_TYPES = (object,)

# ==========================================
# 【显卡分配配置】
# ==========================================
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# 【新增救命神药】：禁用 PCIe P2P 直接通信，强制绕道共享内存！绕开主板硬件黑洞！
os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["NCCL_TIMEOUT"] = "60"
# os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "60"

# 🚀 【新增】：开启 NCCL 异步错误处理。一旦某张卡崩溃，立刻通知全局拉闸，防止死锁卡死！
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

# # 【终极杀手锏：打通网络死结】
# # 强制 NCCL 只能使用名为 eno1 的物理主网卡进行通信，屏蔽所有 Docker 虚拟网卡的干扰！
# os.environ["NCCL_SOCKET_IFNAME"] = "eno1"
# # 禁用 InfiniBand 寻找（你的机器没有，禁用可防干扰）
# os.environ["NCCL_IB_DISABLE"] = "1"

# 导入必要的标准库和第三方库
import re               
import json             
import random   
import numpy as np        
import logging          
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig  
from trl import GRPOConfig, GRPOTrainer  
from datasets import Dataset             
from pymongo import MongoClient          
from peft import LoraConfig, get_peft_model  # 【新增导入】：get_peft_model 用于手动包裹 LoRA

# 【新增】：固定随机种子，确保两张卡从 MongoDB 抽到的题目一模一样！
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- 环境设置 (确保能导包 utils) ---
current_dir = os.path.dirname(os.path.abspath(__file__))  
project_root = os.path.dirname(current_dir)               
if project_root not in sys.path:
    sys.path.append(project_root)  

try:
    from utils import config  
except ImportError:
    print("无法导入 utils.config，请确保路径正确")
    sys.exit(1)  

# 配置日志输出格式
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
    # 【新增】：必须显式设定当前进程的默认 GPU，防止 NCCL 迷路
    torch.cuda.set_device(safe_device_id)
    # ==========================================
    # 🚀 【新增】：打碎 PyTorch 懒加载魔咒！
    # 强迫 PyTorch 立即初始化 CUDA 上下文，赶在 vLLM 篡改环境变量之前锁定双卡！
    # ==========================================
    torch.cuda.get_device_properties(safe_device_id)
    torch.tensor([0.0]).to(current_device)
    logging.info(f"🌐 检测到分布式环境，当前进程 {local_rank} 已绑定至 {current_device}")
else:
    current_device = "cuda:0" if visible_gpus > 0 else "cpu"
    model_device_map = "auto"
    logging.info(f"🖥️ 未检测到分布式环境，采用自适应加载，奖励模型绑定至 {current_device}")


# ==========================================
# 0. 环境与模型路径配置
# ==========================================
MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"  
AGENT_RRM_PATH = "/date/sunchengrui/models/Agent-RRM"                               
OUTPUT_DIR = "./qwen-agent-grpo-output"             
TARGET_DATE = "3_1"  

ANALYSIS_PROMPT = """我在进行流量检测,现在我将给你数条json格式的流量数据,这些流量数据中可能存在多个攻击,也可能不存在攻击。
攻击的类型为”内网穿透“”渗透隐蔽信道“”加密通信“。
我希望你能找到最靠前的攻击流量的编号和这个攻击包含连续的几条流量,并返回其攻击类型。

【工具调用严格约束】：
如果你决定调用 `report_suspicious_traffic_tool` 报警，请务必确保传入的 JSON 参数键名拼写绝对准确！
必须严格使用 `is_suspicious` (布尔值)、`suspicious_flows_start` (整数) 和 `suspicious_flows_end` (整数)。绝对不能拼错或加入多余空格（例如绝不能写成 is_s suspicious），否则会导致系统崩溃！

以下是本次需要分析的流量数据：
{flow_data_json}"""

# ==========================================
# 1. 动态数据集构建核心逻辑
# ==========================================
# (此部分与上一版完全一致，省略具体注释保持排版整洁)
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
    for _ in range(num_samples):
        # L = random.randint(20, 40)
        L = 30
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

# ==========================================
# 2. 初始化本地 Agent-RRM 奖励模型
# ==========================================
logging.info(f"正在加载 Agent-RRM 奖励模型至 {current_device} ...")
reward_model_pipe = pipeline(
    "text-generation", model=AGENT_RRM_PATH, torch_dtype=torch.bfloat16, device=current_device 
)

# ==========================================
# 3. 定义奖励函数
# ==========================================
import math
import datetime

def format_reward_func(completions, **kwargs):
    # 【新增：黑盒破壁机】获取当前进程号并强制打印！
    # 只要控制台打出这句话，说明该卡已经完成 Generate！
    rank = os.environ.get("LOCAL_RANK", "0")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[🕒 {current_time}] 🚀 汇报：Rank {rank} (卡 {rank}) 已经完成本轮 Generate，正在进行裁判打分！", flush=True)
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
            rm_output = reward_model_pipe(rm_prompt, max_new_tokens=150, return_full_text=False)[0]['generated_text']
            score_match = re.search(r'<score>\s*([0-9.]+)\s*</score>', rm_output)
            if score_match: rewards.append(float(score_match.group(1)))
            else: rewards.append(0.0) 
        except Exception:
            rewards.append(0.0)     
    return rewards

# ==========================================
# 4. 配置与启动 GRPOTrainer
# ==========================================
def main():
    # ==========================================
    # 【新增】：双重命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="Qwen GRPO 训练脚本")
    parser.add_argument("--use_4bit", action="store_true", help="是否启用 4-bit QLoRA 量化以极致节省显存")
    parser.add_argument("--use_lora", action="store_true", help="是否启用 LoRA 微调 (不开启则为全参数微调)")
    parser.add_argument("--use_vllm", action="store_true", help="是否启用 vLLM 引擎加速生成阶段")
    parser.add_argument("--use_flash_attn", action="store_true", help="是否启用 FlashAttention-2 极速注意力机制")
    args = parser.parse_args()

    # 增加一个成功标志位
    train_success = False
    try:

        # ==========================================
        # 【新增探针】：提前手动触发 NCCL 握手与通信测试！
        # ==========================================
        if is_distributed:
            import torch.distributed as dist
            from datetime import datetime
            
            # 1. 记录准备初始化的时间
            print(f"\n[🕒 {datetime.now().strftime('%H:%M:%S')}] [Rank {local_rank}] 🔌 准备开始 NCCL 底层握手初始化...", flush=True)
            
            # 手动初始化进程组（如果不写这句，HuggingFace Trainer 稍后也会暗中调用它）
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
                
            print(f"[🕒 {datetime.now().strftime('%H:%M:%S')}] [Rank {local_rank}] ✅ NCCL 初始化完成，进程组建立！", flush=True)

            # 2. 发送物理层面的跨卡握手信号（All-Reduce 测试）
            print(f"[🕒 {datetime.now().strftime('%H:%M:%S')}] [Rank {local_rank}] 📡 正在向其他显卡发送测试张量 (Ping)...", flush=True)
            
            # 造一个值为 1.0 的小张量塞进当前显卡
            dummy_tensor = torch.tensor([1.0]).to(current_device)
            
            # All-Reduce 会强制所有卡互相通信，把大家手里的 1.0 加起来。
            # 如果网络不通，代码会死死卡在下面这一行！
            dist.all_reduce(dummy_tensor)
            
            # 如果成功执行到这里，说明双卡物理通信 100% 畅通！
            print(f"[🕒 {datetime.now().strftime('%H:%M:%S')}] [Rank {local_rank}] 🎉 跨卡握手成功！收到同步结果: {dummy_tensor.item()} (应为 2.0)", flush=True)
            print("="*60, flush=True)

        # 【极其重要的安全守卫】：量化模型不能没有 LoRA
        if args.use_4bit and not args.use_lora:
            logging.error("❌ 致命冲突：你开启了 4-bit 量化，但没有开启 LoRA！量化后的权重是冻结且不可直接求导的，必须配合 LoRA 才能训练。请检查参数！")
            sys.exit(1)

        logging.info("正在加载 Qwen3.5-9B 策略模型 ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "left" 
        # # ✅ 【改为】：
        # tokenizer.padding_side = "right"
        
        

        # 动态构建模型加载参数
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": model_device_map,  
            # 【新增】：强制在计算最后的 Logits 时使用 FP32（全精度），避免 BFloat16 截断导致的概率崩塌！
            "low_cpu_mem_usage": True,
        }

        # 【新增】：动态控制 FlashAttention-2
        if args.use_flash_attn:
            logging.info("⚡ 状态：检测到 --use_flash_attn，已开启 FlashAttention-2 加速！")
            model_kwargs["attn_implementation"] = "flash_attention_2"
        else:
            logging.info("🐢 状态：未开启 FlashAttention-2，使用原生注意力机制。")

        if args.use_4bit:
            logging.info("🌟 状态：已开启 4-bit 量化配置...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model_kwargs["quantization_config"] = bnb_config
        else:
            logging.info("🚀 状态：未开启量化，将以原生 bfloat16 全精度加载基础模型...")
        
        # 加载主模型
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)

        # # 【新增终极防线】：将所有容易溢出的 LayerNorm 层强制转为 FP32 全精度！
        # for name, module in model.named_modules():
        #     if "norm" in name.lower() or "ln" in name.lower():
        #         module.to(torch.float32)
        
        # 修复 PAD Token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

        # ==========================================
        # 【核心】：根据参数决定是否手动包裹 LoRA
        # ==========================================
        if args.use_lora:
            logging.info("🌟 状态：检测到 --use_lora，正在手动包裹 LoRA 适配器 (目标层: o_proj, out_proj)...")
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                # target_modules=["o_proj", "out_proj"], # 必须保持避开 C++ qkv 算子！
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",   # 注意力层
                    "gate_proj", "up_proj", "down_proj",         # FFN 层
                ],
                task_type="CAUSAL_LM",
            )
            # # 手动包裹模型，绕过 Hugging Face Trainer 的内部切分 Bug
            # model = get_peft_model(model, peft_config)
            # 激活输入层的梯度求导 (梯度检查点必须要求)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            # model.print_trainable_parameters()
        else:
            peft_config = None # 如果不开启 LoRA，必须定义为空
            logging.warning("⚠️ 警告：未开启 LoRA！模型将进行【全参数微调】。请密切关注显存是否 OOM！")
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

        # 数据集准备 (Debug 时可以把 num_samples 临时改小)
        dataset = build_grpo_dataset(target_date=TARGET_DATE, range_start=1, range_end=3000, num_samples=200)

        # ==========================================
        # 【新增】：动态构建 vLLM 参数字典
        # ==========================================
        vllm_config = {}
        if args.use_vllm:
            logging.info("🚀 状态：检测到 --use_vllm，已唤醒 vLLM 极速生成引擎！")
            vllm_config = {
                "use_vllm": True,
                "vllm_device": current_device,
                "vllm_gpu_memory_utilization": 0.7, # 划拨 35% 显存给 vLLM
                "vllm_max_model_len": 4096,
            }
        else:
            logging.info("🚶 状态：未开启 vLLM，使用 PyTorch 原生 Generate。")
            vllm_config = {
                "use_vllm": False,
            }

        # 声明 GRPO 全局参数
        training_args = GRPOConfig(
            output_dir=OUTPUT_DIR,            
            learning_rate=5e-6,               
            max_grad_norm=1.0,
            lr_scheduler_type="cosine",       
            logging_steps=1, 
            max_steps=5,                     
            per_device_train_batch_size=1,    
            gradient_accumulation_steps=4,    

            num_generations=4,      
            max_completion_length=512,  
            bf16=True,                        
            gradient_checkpointing=True,      
            report_to="none",                 
            temperature=0.9,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.05,
            
            # 【新增】：将上面构建好的 vLLM 配置动态解包注入
            **vllm_config
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward_func, correctness_reward_func, logic_reward_func],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,  # ✅ 【关键改动】：把 peft_config 重新传给 Trainer！
            # peft_config 已经手动包裹过了，这里不能再传给 Trainer
        )

        logging.info("🚀 开始 GRPO 强化学习训练...")
        trainer.train()  
        trainer.save_model(os.path.join(OUTPUT_DIR, "final_agent_model")) 
        logging.info("✅ 训练完成，模型已保存！")
        train_success = True  # 如果能走到这里，说明没报错
    except Exception as e:
        # 捕捉异常并打印，方便你 Debug
        logging.error(f"❌ 训练过程中发生异常崩溃: {e}")
        raise e
        
    finally:
        if is_distributed:
            import torch.distributed as dist
            if dist.is_initialized():
                # 【关键修改】：只有在训练成功时，才优雅地销毁 NCCL。报错时直接让系统强杀！
                if train_success:
                    logging.info(f"🧹 [Rank {local_rank}] 正在安全销毁 NCCL 进程组...")
                    dist.destroy_process_group()
                    logging.info(f"✅ [Rank {local_rank}] 资源释放完毕，安全退出！")
                else:
                    logging.warning(f"⚠️ [Rank {local_rank}] 发生异常退出，放弃优雅销毁 NCCL 进程组，交由系统回收资源！")

if __name__ == "__main__":
    main()


# 全精度 + LoRA
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py --use_lora
# 4-bit量化 + LoRA
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py --use_4bit --use_lora
# 全参数微调
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py
# 4-bit量化 + 拒绝开启 LoRA
# CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_grpo.py --use_4bit