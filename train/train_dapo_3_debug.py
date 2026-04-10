import os
import sys
import argparse         
import torch            
# 开启 PyTorch 异常侦测（仅用于 Debug，定位具体哪个层溢出）
# torch.autograd.set_detect_anomaly(True)
import requests         
from tqdm import tqdm   # 🚀 加回进度条
from nan_hunter import NaNHunterCallback

class NumericalAuditHook:
    def __init__(self):
        self.log = []

    def hook_fn(self, name, type='fw'):
        def fn(module, input, output):
            # 这里的 output 是本层的激活值（正向）或梯度（反向）
            data = output[0] if isinstance(output, tuple) else output
            if isinstance(data, torch.Tensor):
                stats = {
                    "layer": name,
                    "type": type,
                    "dtype": data.dtype,
                    "max": data.abs().max().item(),
                    "mean": data.mean().item(),
                    "std": data.std().item(),
                }
                # 实时监控：如果梯度超过阈值，立刻报警
                if type == 'bw' and stats['max'] > 1000:
                    print(f"⚠️ [数值爆炸预警] 层: {name} | 梯度 Max: {stats['max']:.2f} | Dtype: {data.dtype}")
                self.log.append(stats)
        return fn

    def attach(self, model):
        for name, module in model.named_modules():
            # 我们重点看注意力输出和 Linear 层，这是误差累加最严重的地方
            if "self_attn" in name or "v_proj" in name or "lm_head" in name:
                module.register_forward_hook(self.hook_fn(name, 'fw'))
                module.register_full_backward_hook(self.hook_fn(name, 'bw'))

class ExplosionHunter:
    def __init__(self, threshold=1000.0):
        self.threshold = threshold
        self.exploded = False

    def hook_fn(self, module_name):
        def bwd_hook(module, grad_input, grad_output):
            if self.exploded:
                return
            
            # 1. 检查从上一层（更靠近 Loss 的顶层）传进来的梯度
            for i, g in enumerate(grad_output):
                if g is not None:
                    max_g = g.abs().max().item()
                    if max_g > self.threshold or torch.isnan(g).any() or torch.isinf(g).any():
                        print(f"\n💥 [爆点定位] 梯度在传入 【{module_name}】 之前就已经异常！")
                        print(f"   传入梯度最大值: {max_g}")
                        self.exploded = True
                        return

            # 2. 检查经过当前层计算后，准备传给下一层的梯度
            for i, g in enumerate(grad_input):
                if g is not None:
                    max_g = g.abs().max().item()
                    if max_g > self.threshold or torch.isnan(g).any() or torch.isinf(g).any():
                        print(f"\n💥 [爆点定位] 梯度在经过 【{module_name}】 的反向计算后瞬间爆炸！")
                        print(f"   传出梯度最大值: {max_g}")
                        self.exploded = True
                        return
        return bwd_hook

    def attach(self, model):
        count = 0
        for name, module in model.named_modules():
            # 过滤掉容器层，只在最底层的算子（如 Linear, self_attn 等）挂载钩子
            if len(list(module.children())) == 0: 
                module.register_full_backward_hook(self.hook_fn(name))
                count += 1
        print(f"🔬 [ExplosionHunter] 已在 {count} 个底层模块挂载反向拦截钩子，等待捕捉爆炸瞬间...")

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

# ==========================================
# 引入真实的 tools 配置
# ==========================================
from tools.tools import tools

def get_tool_schemas(tools_list):
    """提取 Langchain tool 的 JSON Schema 结构"""
    schemas = []
    for t in tools_list:
        # 兼容 Pydantic v1 / v2 的 schema 获取
        if hasattr(t, "args_schema") and t.args_schema:
            if hasattr(t.args_schema, "model_json_schema"):
                params = t.args_schema.model_json_schema()
            else:
                params = t.args_schema.schema()
        else:
            params = {"type": "object", "properties": {}}
        
        # 移除 langchain 特有的可能会干扰模型注意力的 title 字段
        if "title" in params:
            del params["title"]
            
        schemas.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": params
            }
        })
    return schemas

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DAPO] - %(message)s')

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

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"  
AGENT_RRM_PATH = "/date/sunchengrui/models/Agent-RRM"                               
OUTPUT_DIR = "./qwen-agent-dapo-output"             
TARGET_DATE = "3_1"  

RRM_API_URL = "http://localhost:8123/v1/completions"

import math
import datetime

def format_reward_func(completions, **kwargs):
    # 挑第一个生成的文本打印前 500 个字符和最后 200 个字符看看
    print(">>> Debug Start:", completions[0][:500], "...\n... Debug End:", completions[0][-200:])
    rank = os.environ.get("LOCAL_RANK", "0")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n[🕒 {current_time}] 🚀 汇报：Rank {rank} 已经完成本轮 Generate，正在请求 API 裁判打分！", flush=True)
    
    rewards = []
    for comp in completions:
        # 🚀 核心修改：不再匹配 <think>，而是检查是否生成了完整的 XML 工具调用标签
        if "<tool_call>" in comp and "</tool_call>" in comp:
            rewards.append(1.0)
        else:
            rewards.append(-1.0) 
    return rewards

def correctness_reward_func(completions, ground_truth, **kwargs):
    rewards = []
    
    # 仅为了打印一次表头
    printed_debug_this_batch = False
    
    for comp, truth in zip(completions, ground_truth):
        true_start, true_end = truth.get("start"), truth.get("end")
        flow_labels = truth.get("flow_labels", []) 
        
        # 1. 提取参数
        is_susp_match = re.search(r'is_suspicious[\"\']?(?:>|\s*[:=])\s*(True|False|true|false)', comp, re.IGNORECASE)
        start_match = re.search(r'suspicious_flows_start[\"\']?(?:>|\s*[:=])\s*(-?\d+)', comp)
        duration_match = re.search(r'suspicious_flows_duration[\"\']?(?:>|\s*[:=])\s*(-?\d+)', comp)

        # 解析模型判定
        pred_is_suspicious = False
        if is_susp_match and is_susp_match.group(1).lower() == 'true':
            pred_is_suspicious = True
            
        pred_start_val = int(start_match.group(1)) if start_match else None
        pred_duration_val = int(duration_match.group(1)) if duration_match else None

        # 🚀 【新增：打印核对】只挑当前 batch 的第一个生成结果打印，防止刷屏
        if not printed_debug_this_batch:
            print("\n" + "="*50)
            print(f"🎯 [正确率核对] 真实攻击区间: {true_start} -> {true_end}")
            print(f"🤖 [模型预测解析] 是否攻击: {pred_is_suspicious} | Start: {pred_start_val} | Duration: {pred_duration_val}")
            print("="*50 + "\n")
            printed_debug_this_batch = True

        # ==========================================
        # 🛡️ 情况 A：真实情况是没有攻击（正常流量）
        # ==========================================
        if true_start is None:
            if is_susp_match and not pred_is_suspicious:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
            continue

        # ==========================================
        # ⚔️ 情况 B：真实情况是有攻击
        # ==========================================
        if not pred_is_suspicious:
            rewards.append(-1.0)
            continue
            
        if not (start_match and duration_match):
            rewards.append(-1.0)
            continue
            
        pred_start = int(start_match.group(1))
        pred_duration = int(duration_match.group(1))
        
        if pred_start < 1 or pred_duration < 1:
            rewards.append(-1.0)
            continue
            
        # 🚀 核心换算
        pred_end = pred_start + pred_duration - 1
        
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
        score = 0.8 * iou
        rewards.append(score)
        
    return rewards

import concurrent.futures
import re
import logging

def logic_reward_func(completions, **kwargs):
    """带 Debug 打印功能的并发逻辑奖励函数 (模拟随机打分)"""
    rewards = [0.0] * len(completions)
    
    # 局部变量，确保每个 batch (step) 只打印一次，防止控制台刷屏
    printed_this_step = False 

    def evaluate_single(idx, comp):
        nonlocal printed_this_step
        
        # 模拟生成 0.0 到 1.0 之间的随机分数
        random_score = random.uniform(0.0, 1.0)
        
        try:
            # 🚀 【核心排查点】：只打印当前 batch 的第一个线程的日志
            if idx == 0 and not printed_this_step:
                print("\n" + "🌟"*20 + " RRM 裁判 Debug " + "🌟"*20)
                print(f"🤖 [RRM 裁判原始返回]:\n模拟返回分数: {random_score:.2f}\n")
                print(f"✅ [成功提取分数]: {random_score:.2f}")
                print("🌟"*56 + "\n")
                printed_this_step = True
            
            return float(random_score)
                
        except Exception as e:
            if idx == 0 and not printed_this_step:
                print(f"\n❌ [RRM 未知异常]: {e}")
                printed_this_step = True
            return 0.0 

    # 使用多线程瞬间并发发出所有请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(completions)) as executor:
        futures = {
            executor.submit(evaluate_single, i, completions[i]): i 
            for i in range(len(completions))
        }
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            try:
                rewards[idx] = future.result()
            except Exception:
                rewards[idx] = 0.0     
                
    return rewards

def main():
    parser = argparse.ArgumentParser(description="Qwen DAPO 训练脚本")
    parser.add_argument("--use_4bit", action="store_true", help="是否启用 4-bit QLoRA")
    parser.add_argument("--use_lora", action="store_true", help="是否启用 LoRA")
    parser.add_argument("--use_vllm", action="store_true", help="是否启用 vLLM")
    parser.add_argument("--use_flash_attn", action="store_true", help="是否启用 FA2")
    
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

        auditor = NumericalAuditHook()
        auditor.attach(model)

        hunter = ExplosionHunter(threshold=10000.0) # 设定1万为爆炸阈值
        hunter.attach(model)

        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None

        # 同步修正模型的 config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        # 🚀 彻底堵死漏洞：同步修正模型的 generation_config
        if hasattr(model, "generation_config"):
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id

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

        # ==========================================
        # 🚀 改动点：从本地目录读取 Dataset (请确保 export_debug_data.py 已成功执行)
        # ==========================================
        DATASET_CACHE_DIR = "./debug_dataset_cache"
        if not os.path.exists(DATASET_CACHE_DIR):
            logging.error(f"❌ 找不到本地缓存数据，请先运行 export_debug_data.py！")
            sys.exit(1)
        
        logging.info(f"📦 正在从 {DATASET_CACHE_DIR} 加载本地数据集...")
        dataset = Dataset.load_from_disk(DATASET_CACHE_DIR)

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
            learning_rate=5e-9,     
            max_grad_norm=0.1,
            beta=0.04,
            lr_scheduler_type="cosine",       
            logging_steps=1, 

            loss_type="dapo",                   
            epsilon=0.2,                        
            epsilon_high=0.28,                  
            mask_truncated_completions=True,    
            delta=10.0,           
            
            max_steps=3,                     
            per_device_train_batch_size=1,    
            gradient_accumulation_steps=4,    
            
            num_generations=4,      
            max_completion_length=4096,  
            bf16=True,                        
            gradient_checkpointing=True,      
            report_to="tensorboard",
            logging_dir="./runs/qwen-dapo-logs",               
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1,
            optim="adamw_torch",
            
            save_strategy="steps",
            save_steps=1,             
            save_total_limit=3,        

            deepspeed="ds_config_dapo.json",
            
            **vllm_config
        )

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[format_reward_func, correctness_reward_func, logic_reward_func],
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,  
            callbacks=[MemoryCleanupCallback(), NaNHunterCallback()] 
        )

        resume_checkpoint = None
        if args.resume is not None:
            if args.resume == 'True':
                resume_checkpoint = True
                logging.info(f"🔄 自动模式：正尝试从 {OUTPUT_DIR} 寻找最新的 Checkpoint...")
            else:
                resume_checkpoint = args.resume
                logging.info(f"🔄 手动模式：正从指定路径恢复: {resume_checkpoint}")

        logging.info("🚀 开始 DAPO 强化学习训练...")
        
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