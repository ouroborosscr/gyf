import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

def verify_dapo_drift():
    print("\n🔬 [实验 2] 模拟 DAPO 环境下的梯度漂移")
    L, D = 8192, 128
    device, dtype = "cuda", torch.bfloat16
    beta = 0.1 # 你的代码配置

    q = torch.randn(1, L, 1, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, L, 1, D, device=device, dtype=dtype)
    v = torch.randn(1, L, 1, D, device=device, dtype=dtype)

    # --- 步骤 1: 用 FA2 计算作为“基准” ---
    out_fa2 = flash_attn_func(q, k, v, causal=True)
    # 模拟 DAPO 的 LogProb 提取
    log_pi_fa2 = out_fa2.mean() 
    log_pi_fa2.backward(retain_graph=True)
    grad_fa2 = q.grad.clone()
    q.grad.zero_()

    # --- 步骤 2: 用 SDPA Math 计算 ---
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_sdpa = F.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=True
        ).transpose(1,2)
    
    log_pi_sdpa = out_sdpa.mean()
    log_pi_sdpa.backward()
    grad_sdpa = q.grad.clone()

    # --- 步骤 3: 模拟 DAPO 指数项的影响 ---
    # DAPO 的梯度包含 exp(log_pi - log_ref)
    # 如果 log_pi 有微小偏差 (假设偏差为 0.1)
    error_multiplier = torch.exp(torch.tensor(0.1 * beta))
    
    print(f"    SDPA 与 FA2 梯度的余弦相似度: {F.cosine_similarity(grad_fa2.flatten(), grad_sdpa.flatten(), dim=0).item():.6f}")
    print(f"    SDPA 梯度的最大范数: {grad_sdpa.abs().max().item():.10f}")
    print(f"    FA2  梯度的最大范数: {grad_fa2.abs().max().item():.10f}")
    print(f"    💡 结论：如果在长序列下相似度下降，说明 SDPA Math 产生了不稳定的梯度方向。")

if __name__ == "__main__":
    verify_dapo_drift()