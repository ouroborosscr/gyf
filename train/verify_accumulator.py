import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

def verify_fp32_accumulator():
    print("="*60)
    print("🔍 [决定性证据] 验证算子底层累加器真实精度")
    print("="*60)

    B, L, H, D = 1, 4096, 1, 128
    device, dtype = "cuda", torch.bfloat16

    # 初始化确定性的 Q, K, V
    q = torch.zeros(B, L, H, D, device=device, dtype=dtype)
    k = torch.zeros(B, L, H, D, device=device, dtype=dtype)
    v = torch.zeros(B, L, H, D, device=device, dtype=dtype)

    # 构造数学陷阱：
    # Q * K^T / sqrt(128) = K[..., 0]
    q[0, 0, 0, 0] = 128 ** 0.5  # 消除缩放因子的影响
    
    # K 的第一个元素为 5.0，exp(5.0) ≈ 148.4
    k[0, 0, 0, 0] = 5.0
    # K 剩下的 4095 个元素为 -5.0，exp(-5.0) ≈ 0.0067
    k[0, 1:, 0, 0] = -5.0
    
    # V 第一个元素为 1.0，其余为 0.0
    v[0, 0, 0, 0] = 1.0

    # 理论推导：
    # 如果用 FP32 累加 Softmax 分母：148.4 + (4095 * 0.0067) ≈ 175.8
    # 最终输出的 V = 148.4 / 175.8 * 1.0 ≈ 0.84
    # 如果用 BF16 累加 Softmax 分母：148.4 + 0.0067 每次都被截断，分母永远是 148.4
    # 最终输出的 V = 148.4 / 148.4 * 1.0 = 1.00

    # 1. 运行 FA2
    out_fa2 = flash_attn_func(q, k, v, causal=False)
    
    # 2. 运行 SDPA (Math)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k.transpose(1, 2)
    v_sdpa = v.transpose(1, 2)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa).transpose(1, 2)

    # 3. 运行纯 FP32 理想计算 (真理基准)
    q32, k32, v32 = q_sdpa.float(), k_sdpa.float(), v_sdpa.float()
    scores = torch.matmul(q32, k32.transpose(-1, -2)) / (128**0.5)
    probs = torch.softmax(scores, dim=-1)
    out_fp32 = torch.matmul(probs, v32).transpose(1, 2)

    print(f"🎯 纯 FP32 理论基准值: {out_fp32[0, 0, 0, 0].item():.4f} (期望约为 0.84)")
    print(f"🚀 FA2 输出结果: {out_fa2[0, 0, 0, 0].item():.4f}")
    print(f"🐢 SDPA Math 输出结果: {out_sdpa[0, 0, 0, 0].item():.4f}")
    
    print("\n💡 [结论判定]:")
    print("   如果 FA2 接近 0.84，证明它底层确实用了 FP32 寄存器来保护累加。")
    print("   如果 SDPA 接近 1.00，证明它在 BF16 模式下发生了严重的截断坍塌（大数吃小数）。")

if __name__ == "__main__":
    verify_fp32_accumulator()