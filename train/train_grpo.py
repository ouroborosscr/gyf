import os

# ==========================================
# 【显卡分配配置】
# 限制程序只能看到物理机上的 0 号和 1 号显卡
# 注意：这行代码必须在 import torch 等操作之前执行！
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
# ==========================================

import re
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ==========================================
# 0. 环境与模型路径配置
# ==========================================
MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"  # 你的基础策略模型路径
AGENT_RRM_PATH = "../Reagent"                      # 过程奖励模型路径 (本地路径)
OUTPUT_DIR = "./qwen-agent-grpo-output"

# ==========================================
# 1. 初始化本地 Agent-RRM 奖励模型
# ==========================================
print("正在加载 Agent-RRM 奖励模型至 cuda:0 ...")
# 强制让奖励模型加载到分配好的 0 号卡上，避免与大模型抢占显存
reward_model_pipe = pipeline(
    "text-generation", 
    model=AGENT_RRM_PATH, 
    torch_dtype=torch.bfloat16, 
    device="cuda:0"  
)

# ==========================================
# 2. 定义奖励函数 (Reward Functions)
# ==========================================

def format_reward_func(completions, **kwargs):
    """
    格式奖励：强制模型必须输出 <think>...</think> 过程
    """
    rewards = []
    for comp in completions:
        # 匹配是否包含完整的思考标签
        if re.search(r"<think>.*?</think>", comp, re.DOTALL):
            rewards.append(1.0)
        else:
            rewards.append(-1.0) # 没思考直接作答，严重扣分
    return rewards

def correctness_reward_func(completions, ground_truth, **kwargs):
    """
    结果奖励 (ORM)：检查提取的可疑流量起始和终止编号是否与真实标签一致
    """
    rewards = []
    for comp, truth in zip(completions, ground_truth):
        score = 0.0
        # 获取真实的起始和终止位置
        true_start = truth.get("start")
        true_end = truth.get("end")

        # 用正则从模型的输出中粗略提取 start 和 end 的数字
        start_match = re.search(r'(?:start|suspicious_flows_start)[\"\']?\s*[:=]\s*(\d+)', comp)
        end_match = re.search(r'(?:end|suspicious_flows_end)[\"\']?\s*[:=]\s*(\d+)', comp)

        if start_match and end_match:
            pred_start = int(start_match.group(1))
            pred_end = int(end_match.group(1))
            
            # 精准命中
            if pred_start == true_start and pred_end == true_end:
                score = 1.0
            # 命中了一部分（存在交集）
            elif max(pred_start, true_start) <= min(pred_end, true_end):
                score = 0.5
            # 完全报假警 (False Positive)
            else:
                score = -0.5
        else:
            # 没有调用工具或格式错误
            if true_start is None:
                score = 1.0  # 真实情况也没攻击，模型也没报，得分
            else:
                score = -1.0 # 有攻击但漏报了 (False Negative)

        rewards.append(score)
    return rewards

def logic_reward_func(completions, **kwargs):
    """
    过程奖励 (PRM)：使用 Agent-RRM 评估思考过程与工具调用之间的逻辑连贯性
    """
    rewards = []
    for comp in completions:
        # 构造发给 Agent-RRM 的 Prompt
        rm_prompt = f"Please evaluate the following agent reasoning and action:\n{comp}\n\n"
        
        try:
            # 调用部署在 cuda:0 的奖励模型生成打分
            rm_output = reward_model_pipe(
                rm_prompt, 
                max_new_tokens=150, 
                return_full_text=False
            )[0]['generated_text']
            
            # 从 Agent-RRM 的输出中提取 <score> 标签里的浮点数字
            score_match = re.search(r'<score>\s*([0-9.]+)\s*</score>', rm_output)
            if score_match:
                score = float(score_match.group(1))
                rewards.append(score)
            else:
                rewards.append(0.0)
        except Exception as e:
            # 推理失败默认给 0 分
            rewards.append(0.0)
            
    return rewards

# ==========================================
# 3. 准备数据集 (Dataset)
# ==========================================
def get_dummy_dataset():
    """
    占位数据集。在实际训练时，请读取你通过 GYF_Test.py 收集到的流量文本
    并将真实的攻击区间填入 ground_truth。
    """
    data = {
        "prompt": [
            "分析以下 30 条流量，找出攻击流量的编号。流量数据：[{...}]",
            "分析以下 30 条流量，找出攻击流量的编号。流量数据：[{...}]"
        ],
        "ground_truth": [
            {"start": 18, "end": 20},
            {"start": 1, "end": 14}
        ]
    }
    return Dataset.from_dict(data)

# ==========================================
# 4. 配置与启动 GRPOTrainer
# ==========================================
def main():
    print("正在加载 Qwen3.5-9B 策略模型至 cuda:1 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # Qwen 系列如果没设 pad_token，通常指定为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 强制让被训练的大模型加载到分配好的 1 号卡上
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map={"": "cuda:1"} 
    )

    dataset = get_dummy_dataset()

    # GRPO 训练参数配置
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        logging_steps=10,
        max_steps=500,                    
        per_device_train_batch_size=1,    # 每个设备的 batch_size
        gradient_accumulation_steps=4,
        num_generations=4,                # G_K: 每个 prompt 采样 4 个不同的答案用于优势估计
        max_prompt_length=2048,           # 输入的 prompt 最大长度
        max_completion_length=512,        # 预留给 <think> 过程的最大长度
        bf16=True,                        
        report_to="none",                 
    )

    # 组装 Trainer 并传入三种奖励函数
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func, 
            correctness_reward_func, 
            logic_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("🚀 开始 GRPO 强化学习训练...")
    trainer.train()
    
    # 保存最终对齐后的模型
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_agent_model"))
    print("✅ 训练完成，模型已保存！")

if __name__ == "__main__":
    main()