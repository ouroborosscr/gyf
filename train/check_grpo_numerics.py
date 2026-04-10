"""
检查 TRL GRPOTrainer 的 GRPO loss 计算路径中每个环节的数值保护
用法: python check_grpo_numerics.py
"""
import inspect
import textwrap

print("=" * 70)
print("检查 TRL GRPOTrainer 的数值保护")
print("=" * 70)

from trl.trainer.grpo_trainer import GRPOTrainer
from trl import GRPOConfig

# ============================================================
print("\n" + "=" * 70)
print("1. _compute_loss / _compute_loss 方法 - GRPO loss 核心")
print("=" * 70)

if hasattr(GRPOTrainer, '_compute_loss'):
    src = inspect.getsource(GRPOTrainer._compute_loss)
    print(src)
else:
    print("没有 _compute_loss 方法")
    # 尝试 compute_loss
    if hasattr(GRPOTrainer, 'compute_loss'):
        src = inspect.getsource(GRPOTrainer.compute_loss)
        print(src[:3000])

# ============================================================
print("\n" + "=" * 70)
print("2. _get_per_token_logps_and_entropies - log probability 计算")
print("=" * 70)

if hasattr(GRPOTrainer, '_get_per_token_logps_and_entropies'):
    src = inspect.getsource(GRPOTrainer._get_per_token_logps_and_entropies)
    print(src)

# ============================================================
print("\n" + "=" * 70)
print("3. _generate_and_score_completions - advantage 计算")
print("=" * 70)

if hasattr(GRPOTrainer, '_generate_and_score_completions'):
    src = inspect.getsource(GRPOTrainer._generate_and_score_completions)
    # 只打印 advantage 相关部分
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['advantage', 'mean_grouped', 'std_grouped', 'reward', 'normalize']):
            start = max(0, i-2)
            end = min(len(lines), i+3)
            for j in range(start, end):
                print(f"  {j:>4}: {lines[j]}")
            print("  ...")

# ============================================================
print("\n" + "=" * 70)
print("4. GRPOConfig 默认参数检查")
print("=" * 70)

# 检查关键参数的默认值
config = GRPOConfig(output_dir="/tmp/test")
checks = {
    "beta (KL 系数)": getattr(config, 'beta', 'N/A'),
    "max_grad_norm": getattr(config, 'max_grad_norm', 'N/A'),
    "loss_type": getattr(config, 'loss_type', 'N/A'),
    "max_completion_length": getattr(config, 'max_completion_length', 'N/A'),
    "num_generations": getattr(config, 'num_generations', 'N/A'),
    "epsilon (clip ratio)": getattr(config, 'epsilon', getattr(config, 'cliprange', 'N/A')),
    "reward_weights": getattr(config, 'reward_weights', 'N/A'),
    "scale_rewards": getattr(config, 'scale_rewards', 'N/A'),
    "norm_reward": getattr(config, 'norm_reward', getattr(config, 'normalize_reward', 'N/A')),
}
for k, v in checks.items():
    print(f"  {k}: {v}")

# 列出所有 config 属性（找隐藏的保护参数）
print("\n  所有 GRPOConfig 属性中含 clip/norm/scale/eps/clamp 的:")
for attr in sorted(dir(config)):
    if attr.startswith('_'):
        continue
    if any(kw in attr.lower() for kw in ['clip', 'norm', 'scale', 'eps', 'clamp', 'ratio', 'advantage', 'kl']):
        val = getattr(config, attr, None)
        if not callable(val):
            print(f"    {attr} = {val}")

# ============================================================
print("\n" + "=" * 70)
print("5. 逐行检查 _compute_loss 中的数值风险点")
print("=" * 70)

if hasattr(GRPOTrainer, '_compute_loss'):
    src = inspect.getsource(GRPOTrainer._compute_loss)
    lines = src.split('\n')
    
    risk_keywords = {
        'sum': '⚠️  求和（可能未除以序列长度）',
        'exp(': '⚠️  指数运算（可能溢出）',
        'torch.exp': '⚠️  指数运算（可能溢出）',
        '.exp()': '⚠️  指数运算（可能溢出）',
        'log_ratio': '📌 log ratio 计算',
        'ratio': '📌 importance sampling ratio',
        'advantage': '📌 advantage 使用',
        'clamp': '✅ clamp 保护',
        'clip': '✅ clip 保护',
        'mean': '📌 mean 操作',
        'per_token': '📌 per-token 操作',
    }
    
    print("  数值风险点扫描:")
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        for keyword, desc in risk_keywords.items():
            if keyword in stripped.lower():
                print(f"  {i:>4}: {desc}")
                print(f"        {stripped[:120]}")
                break

# ============================================================
print("\n" + "=" * 70)
print("6. 检查 loss_type 和 ratio clipping 实现")
print("=" * 70)

# 搜索 exp 和 clamp 在 loss 计算中的使用
for method_name in ['_compute_loss', 'compute_loss', '_get_per_token_logps_and_entropies']:
    if hasattr(GRPOTrainer, method_name):
        src = inspect.getsource(getattr(GRPOTrainer, method_name))
        
        has_exp = 'exp(' in src or '.exp()' in src
        has_clamp = 'clamp' in src or 'clip' in src
        has_mean_over_tokens = '/= ' in src or 'mean' in src
        has_log_ratio = 'log_ratio' in src or 'logp' in src
        
        print(f"\n  {method_name}:")
        print(f"    包含 exp: {has_exp}")
        print(f"    包含 clamp/clip: {has_clamp}")
        print(f"    包含 mean/除法: {has_mean_over_tokens}")
        print(f"    包含 log_ratio/logp: {has_log_ratio}")

# ============================================================
print("\n" + "=" * 70)
print("7. 模拟 GRPO loss 的数值行为（不同序列长度）")
print("=" * 70)

import torch

for seq_len in [256, 512, 1024, 2048, 4096]:
    # 模拟 per_token_logps (通常在 -5 到 -15 之间)
    per_token_logps = torch.randn(1, seq_len, dtype=torch.bfloat16) * 3 - 8
    ref_per_token_logps = torch.randn(1, seq_len, dtype=torch.bfloat16) * 3 - 8
    
    # log ratio = policy logp - ref logp
    log_ratio = per_token_logps - ref_per_token_logps
    
    # 如果直接 sum
    log_ratio_sum = log_ratio.sum(dim=1)
    
    # 如果 mean
    log_ratio_mean = log_ratio.mean(dim=1)
    
    # exp(log_ratio) 的行为
    ratio_per_token = torch.exp(log_ratio)
    ratio_sum = torch.exp(log_ratio_sum)
    
    print(f"\n  seq_len={seq_len}:")
    print(f"    log_ratio per token: mean={log_ratio.mean():.2f}, std={log_ratio.std():.2f}")
    print(f"    log_ratio SUM: {log_ratio_sum.item():.2f}")
    print(f"    log_ratio MEAN: {log_ratio_mean.item():.4f}")
    print(f"    exp(per_token_ratio) max: {ratio_per_token.max().item():.4f}")
    print(f"    exp(SUM): {ratio_sum.item()}")  # 这个容易爆
    print(f"    nan/inf in exp(SUM): nan={torch.isnan(ratio_sum).any().item()}, inf={torch.isinf(ratio_sum).any().item()}")

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)