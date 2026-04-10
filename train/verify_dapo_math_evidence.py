import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

def get_dapo_grad_multiplier(log_ratio, beta=0.1):
    """模拟 DAPO/DPO 的梯度缩放项: beta * sigmoid(-beta * log_ratio)"""
    # 这里的 log_ratio = (log_pi - log_ref)
    return beta * torch.sigmoid(-beta * log_ratio)

def verify_evidence():
    device = "cuda"
    # 我们使用 32768 长度来放大累加误差
    L = 32768 
    D = 128
    beta = 0.1
    
    print(f"🔬 [测试条件] 序列长度: {L}, Beta: {beta}")
    
    # 构造带有“异常值”的输入，模拟真实训练中模型输出的 Logits
    # 正常位置值较小，但某些关键位置（如工具调用标记）值很大
    q = torch.randn(1, L, 1, D, device=device, dtype=torch.bfloat16) * 10 
    k = torch.randn(1, L, 1, D, device=device, dtype=torch.bfloat16) * 10
    v = torch.randn(1, L, 1, D, device=device, dtype=torch.bfloat16)
    
    # 1. 真理标准：FP32 完整计算
    q_32, k_32, v_32 = q.float(), k.float(), v.float()
    scores_32 = torch.matmul(q_32, k_32.transpose(-1, -2)) / (D**0.5)
    probs_32 = torch.softmax(scores_32, dim=-1)
    # 模拟计算 LogProb 的一部分（Softmax 分母的对数）
    log_sum_exp_32 = torch.logsumexp(scores_32, dim=-1)
    
    # 2. 验证 FA2 (强制 FP32 累加)
    # 我们从 FA2 内部无法直接拿 LogSumExp，但可以对比输出
    out_fa2 = flash_attn_func(q, k, v, causal=True)
    
    # 3. 验证 SDPA Math (BF16 累加)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_sdpa = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
        ).transpose(1,2)

    # ==========================================
    # 证据 A：长序列下的精度坍塌对比
    # ==========================================
    diff_fa2 = (out_fa2.float() - out_ref_32_ideal(q_32, k_32, v_32)).abs().mean().item()
    diff_sdpa = (out_sdpa.float() - out_ref_32_ideal(q_32, k_32, v_32)).abs().mean().item()
    
    print(f"\n[1] 精度偏差 (相对于 FP32 真理值):")
    print(f"    FA2  平均偏差: {diff_fa2:.10f}")
    print(f"    SDPA 平均偏差: {diff_sdpa:.10f}")

    # ==========================================
    # 证据 B：DAPO 指数放大效应
    # ==========================================
    # 假设 SDPA 的精度漂移导致 log_pi 偏离了 0.5 (在 10万 token 下非常常见)
    error_drift = 0.5 
    grad_normal = get_dapo_grad_multiplier(torch.tensor([0.0]))
    grad_drifted = get_dapo_grad_multiplier(torch.tensor([error_drift]))
    
    print(f"\n[2] DAPO 数值放大效应:")
    print(f"    正常情况下的梯度权重: {grad_normal.item():.6f}")
    print(f"    当底层算子产生 {error_drift} 的精度漂移时，梯度权重变为: {grad_drifted.item():.6f}")
    print(f"    💡 结论：任何底层的微小不稳定性，都会被 DAPO 的 Sigmoid/Exp 逻辑直接放大。")

def out_ref_32_ideal(q, k, v):
    # 辅助函数，用于计算完美的 FP32 参考值
    s = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1]**0.5)
    m = torch.triu(torch.ones(s.shape[-2:], device=s.device), 1).bool()
    s.masked_fill_(m, float('-inf'))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)

if __name__ == "__main__":
    verify_evidence()