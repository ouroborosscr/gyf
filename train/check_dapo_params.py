"""
检查 DAPO 参数是否真的被 GRPOConfig 接收到了
用法: python check_dapo_params.py
"""
from trl import GRPOConfig

# 模拟你设置的参数
config = GRPOConfig(
    output_dir="/tmp/test",
    loss_type="dapo",
    learning_rate=5e-9,
    max_grad_norm=0.3,
    beta=0.04,
    # 你应该设了这些：
    epsilon_low=0.2,
    epsilon_high=0.28,
    delta=10.0,
)

print("=" * 50)
print("GRPOConfig 实际值:")
print("=" * 50)

params = {
    "loss_type": config.loss_type,
    "learning_rate": config.learning_rate,
    "max_grad_norm": config.max_grad_norm,
    "beta": config.beta,
    "epsilon": getattr(config, 'epsilon', 'N/A'),
    "epsilon_low": getattr(config, 'epsilon_low', 'N/A'),
    "epsilon_high": getattr(config, 'epsilon_high', 'N/A'),
    "delta": getattr(config, 'delta', 'N/A'),
    "num_iterations": getattr(config, 'num_iterations', 'N/A'),
    "importance_sampling_level": getattr(config, 'importance_sampling_level', 'N/A'),
    "scale_rewards": getattr(config, 'scale_rewards', 'N/A'),
    "gradient_accumulation_steps": config.gradient_accumulation_steps,
}

for k, v in params.items():
    print(f"  {k}: {v}")

# 模拟第一步的 ratio
import torch
print("\n" + "=" * 50)
print("模拟第一步 ratio (old_logps == new_logps)")
print("=" * 50)

seq_len = 2000
log_ratio = torch.zeros(1, seq_len)  # 第一步 log_ratio = 0
coef_1 = torch.exp(log_ratio)  # = 1.0

print(f"  log_ratio: {log_ratio[0,0].item()}")
print(f"  coef_1 (before clamp): {coef_1[0,0].item()}")

# 检查 clamp 是否有效
coef_2 = torch.clamp(coef_1, 1 - 0.2, 1 + 0.28)
print(f"  coef_2 (after clamp [0.8, 1.28]): {coef_2[0,0].item()}")
print(f"  → clamp 对 1.0 无效，ratio 保持 1.0")

if 10.0 is not None:
    coef_1_delta = torch.clamp(coef_1, max=10.0)
    print(f"  coef_1 (after delta clamp ≤10): {coef_1_delta[0,0].item()}")
    print(f"  → delta 对 1.0 也无效")

print(f"\n  结论: 第一步 ratio=1.0，所有 clamp/delta 都不起作用")
print(f"  梯度大小完全取决于 loss 聚合方式")

# 对比 DAPO vs GRPO 的 loss 大小差异
print("\n" + "=" * 50)
print("DAPO vs GRPO 的 loss scale 差异")
print("=" * 50)

advantage = 1.5  # 典型值
per_token_loss = advantage * 1.0  # ratio=1, advantage=1.5
num_tokens = 2000
batch_size = 4
grad_accum = 4

dapo_loss = (per_token_loss * num_tokens) / batch_size
grpo_loss = (per_token_loss * num_tokens / num_tokens) / grad_accum  # mean over tokens, then /accum

print(f"  per_token_loss: {per_token_loss}")
print(f"  序列长度: {num_tokens}")
print(f"  batch_size: {batch_size}")
print(f"  grad_accum: {grad_accum}")
print(f"")
print(f"  DAPO loss = sum / batch = {dapo_loss}")
print(f"  GRPO loss = mean / accum = {grpo_loss}")
print(f"  比值: DAPO / GRPO = {dapo_loss / grpo_loss}x")
print(f"")
print(f"  这个 {dapo_loss / grpo_loss}x 的差异经过 36 层 transformer 反传会指数放大")
print(f"  所以梯度达到 10^27 级别")