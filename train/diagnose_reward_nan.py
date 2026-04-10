"""
诊断：模拟 GRPO 的奖励 → 优势 → 损失 计算链，检查 NaN 来源
用法: LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/python3.11/site-packages/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_reward_nan.py
"""
import torch
import math
import numpy as np

# ============================================================
print("=" * 60)
print("Step 1: 检查三个 reward 函数的输出范围")
print("=" * 60)

# 模拟 correctness_reward_func 的所有分支
def simulate_correctness_rewards():
    """穷举 correctness_reward_func 的各个分支，输出所有可能的值"""
    scores = []
    
    # 情况A: 无攻击 → 1.0 或 -1.0
    scores.extend([1.0, -1.0])
    
    # 情况B: 漏报/缺参数 → -1.0
    scores.append(-1.0)
    
    # 完美匹配 → 1.0
    scores.append(1.0)
    
    # 完全不重叠（pred 在前）: -0.5 + 0.4 * exp(-0.2 * dist)
    for dist in [1, 2, 5, 10, 20, 50]:
        s = -0.5 + 0.4 * math.exp(-0.2 * dist)
        scores.append(s)
    
    # 完全不重叠（pred 在后）: -0.9 + 0.4 * exp(-0.2 * dist)
    for dist in [1, 2, 5, 10, 20, 50]:
        s = -0.9 + 0.4 * math.exp(-0.2 * dist)
        scores.append(s)
    
    # pred 包含 true: 0.5 + 0.4 * exp(-0.2 * penalty)
    for penalty in [0, 1, 2, 5, 10, 25]:
        s = 0.5 + 0.4 * math.exp(-0.2 * penalty)
        scores.append(s)
    
    # true 包含 pred: 0.1 + 0.4 * exp(-0.3 * penalty)
    for penalty in [0, 1, 2, 5, 10, 25]:
        s = 0.1 + 0.4 * math.exp(-0.3 * penalty)
        scores.append(s)
    
    # IoU 部分重合: 0.8 * iou
    for iou in [0.1, 0.3, 0.5, 0.7, 0.9]:
        scores.append(0.8 * iou)
    
    return scores

correctness_scores = simulate_correctness_rewards()
print(f"\ncorrectness_reward_func 可能值范围:")
print(f"  min={min(correctness_scores):.6f}")
print(f"  max={max(correctness_scores):.6f}")
print(f"  所有可能值: {sorted(set(round(s, 6) for s in correctness_scores))}")

# format_reward: 1.0 或 -1.0
# logic_reward: 0.0 到 1.0
print(f"\nformat_reward_func: [-1.0, 1.0]")
print(f"logic_reward_func:  [0.0, 1.0]")

# ============================================================
print("\n" + "=" * 60)
print("Step 2: 模拟 GRPO 的 advantage 计算")
print("=" * 60)

# GRPO 的 advantage 计算:
# 1. 对每组 num_generations 个 completion 的 reward 做 group normalization
# 2. advantage = (reward - mean) / (std + eps)
# 
# 问题场景：当 num_generations=4 个样本的 reward 全部相同时，std=0

def compute_grpo_advantages(rewards_per_group, eps=1e-4):
    """模拟 GRPO 的 group advantage 计算"""
    rewards = np.array(rewards_per_group)
    mean = rewards.mean()
    std = rewards.std()
    advantages = (rewards - mean) / (std + eps)
    return advantages, mean, std

print("\n场景模拟 (num_generations=4):")

# 场景1: 全部相同
test_cases = [
    ("全部 -1.0 (都没格式)", [-1.0, -1.0, -1.0, -1.0]),
    ("全部 1.0 (都有格式)", [1.0, 1.0, 1.0, 1.0]),
    ("3个-1 + 1个1", [-1.0, -1.0, -1.0, 1.0]),
    ("正常分布", [0.3, 0.5, -0.2, 0.8]),
    ("极端差异", [-1.0, -1.0, -1.0, 0.9]),
]

for name, rewards in test_cases:
    adv, mean, std = compute_grpo_advantages(rewards)
    print(f"\n  {name}: rewards={rewards}")
    print(f"    mean={mean:.4f}, std={std:.6f}")
    print(f"    advantages={[f'{a:.4f}' for a in adv]}")
    if std < 1e-6:
        print(f"    ⚠️ std 接近 0! advantage 可能产生极大值")

# ============================================================
print("\n" + "=" * 60)
print("Step 3: 模拟 GRPO 合并三个 reward 后的效果")
print("=" * 60)

# GRPO 会对每个 reward 函数分别计算，然后加权合并
# GRPOTrainer 默认把多个 reward 函数的 reward 相加

print("\n模拟一个 batch (4 个 generations) 的合并 reward:")

# 典型场景: 第一步训练，模型还没学会格式
format_rewards = [-1.0, -1.0, -1.0, -1.0]  # 都没格式
correct_rewards = [-1.0, -1.0, -1.0, -1.0]  # 都判错
logic_rewards = [0.75, 0.85, 0.65, 0.80]    # RRM 给的分

combined = [f + c + l for f, c, l in zip(format_rewards, correct_rewards, logic_rewards)]
print(f"  format:    {format_rewards}")
print(f"  correct:   {correct_rewards}")
print(f"  logic:     {logic_rewards}")
print(f"  combined:  {combined}")

adv, mean, std = compute_grpo_advantages(combined)
print(f"  mean={mean:.4f}, std={std:.6f}")
print(f"  advantages: {[f'{a:.4f}' for a in adv]}")

# 更危险的场景：所有 reward 都完全相同
print("\n最危险场景: 三个函数返回值合并后完全相同:")
combined_same = [-1.25, -1.25, -1.25, -1.25]
adv2, mean2, std2 = compute_grpo_advantages(combined_same)
print(f"  combined={combined_same}")
print(f"  mean={mean2:.4f}, std={std2:.10f}")
print(f"  advantages: {[f'{a:.4f}' for a in adv2]}")
print(f"  ⚠️ 当 std=0 时 advantage 全为 0，GRPO loss 退化")

# ============================================================
print("\n" + "=" * 60)
print("Step 4: 模拟 GRPO loss 中的 per_token_logps 数值范围")
print("=" * 60)

# GRPO loss = -advantage * (policy_logps - ref_logps) + beta * KL
# per_token_logps 是每个 token 的 log probability
# 对于长序列，序列级别的 logps (sum) 会非常大

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 模拟不同序列长度下 per_token_logps 的累积
for seq_len in [512, 1024, 2048, 4096]:
    # 典型的 per-token log prob 在 -2 到 -10 之间
    fake_token_logps = torch.randn(1, seq_len, device=device, dtype=torch.bfloat16) * 3 - 5
    
    sum_logps = fake_token_logps.sum()
    mean_logps = fake_token_logps.mean()
    
    # 模拟 policy - ref 的差
    fake_diff = torch.randn(1, seq_len, device=device, dtype=torch.bfloat16) * 0.1
    
    # 乘以 advantage
    advantage = torch.tensor([5.0], device=device, dtype=torch.bfloat16)  # 极端 advantage
    weighted = advantage * fake_diff.sum()
    
    print(f"  seq_len={seq_len:>5}: sum_logps={sum_logps.item():.1f}, "
          f"mean_logps={mean_logps.item():.4f}, "
          f"weighted_loss={weighted.item():.2f}")
    
    # 检查 bf16 是否溢出
    if torch.isinf(weighted) or torch.isnan(weighted):
        print(f"    ⚠️ bf16 溢出!")

# ============================================================
print("\n" + "=" * 60)
print("Step 5: 检查 bf16 下 exp 和 log 的数值边界")
print("=" * 60)

# GRPO 内部会做 log_softmax 和 exp，bf16 的范围有限
bf16_max = torch.finfo(torch.bfloat16).max
bf16_min = torch.finfo(torch.bfloat16).min
bf16_tiny = torch.finfo(torch.bfloat16).tiny

print(f"bf16 max:  {bf16_max}")
print(f"bf16 min:  {bf16_min}")
print(f"bf16 tiny: {bf16_tiny}")

# log(tiny) in bf16
log_tiny = torch.tensor(bf16_tiny, dtype=torch.bfloat16).log()
print(f"log(bf16_tiny) = {log_tiny.item()}")

# 当 log prob 非常负时，exp(logp) 会下溢为 0
test_logps = torch.tensor([-10, -50, -100, -200, -500], dtype=torch.bfloat16)
exp_logps = test_logps.exp()
print(f"\nexp(logps) in bf16:")
for lp, ep in zip(test_logps.tolist(), exp_logps.tolist()):
    print(f"  exp({lp:.0f}) = {ep}")

# 当 advantage 乘以极端 logps 差值时
print(f"\n极端情况: advantage * sum(logps_diff) in bf16:")
for adv_val in [1.0, 5.0, 10.0, 50.0]:
    for diff_sum in [10.0, 100.0, 1000.0, 5000.0]:
        result = torch.tensor(adv_val, dtype=torch.bfloat16) * torch.tensor(diff_sum, dtype=torch.bfloat16)
        flag = "⚠️" if (torch.isinf(result) or torch.isnan(result)) else "  "
        print(f"  {flag} {adv_val:.0f} × {diff_sum:.0f} = {result.item()}")

# ============================================================
print("\n" + "=" * 60)
print("Step 6: 用真实模型的 logits 测试 (短序列)")
print("=" * 60)

if torch.cuda.is_available():
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None
    
    # 用真实 token 做 forward，检查 logits 和 log_softmax
    for seq_len in [256, 1024, 2048]:
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids)
        torch.cuda.synchronize()
        
        logits = out.logits  # [1, seq_len, vocab_size]
        
        # 计算 per-token log probs (和 GRPO 内部做的一样)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # 取每个 token 对应的 log prob
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        shift_log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        per_token_logps = shift_log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        print(f"\n  seq_len={seq_len}:")
        print(f"    logits  - min={logits.min().item():.2f}, max={logits.max().item():.2f}, "
              f"nan={torch.isnan(logits).any().item()}, inf={torch.isinf(logits).any().item()}")
        print(f"    log_softmax - nan={torch.isnan(log_probs).any().item()}, inf={torch.isinf(log_probs).any().item()}")
        print(f"    per_token_logps - min={per_token_logps.min().item():.2f}, max={per_token_logps.max().item():.2f}, "
              f"mean={per_token_logps.mean().item():.2f}")
        print(f"    per_token_logps sum={per_token_logps.sum().item():.1f}")
        print(f"    nan={torch.isnan(per_token_logps).any().item()}, inf={torch.isinf(per_token_logps).any().item()}")
        
        torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)