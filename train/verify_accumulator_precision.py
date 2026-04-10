import torch
from flash_attn import flash_attn_func

def test_accumulator():
    print("🔬 [实验 1] 验证算子内部累加器精度 (Epsilon Test)")
    L, D = 4096, 128
    device, dtype = "cuda", torch.bfloat16

    # 1. 构造一个背景基数 A
    q = torch.zeros(1, L, 1, D, device=device, dtype=dtype)
    k = torch.zeros(1, L, 1, D, device=device, dtype=dtype)
    # 让第一个 token 的点积刚好等于 1.0 (经过转换后)
    q[0, 0, 0, 0] = 1.0
    k[0, 0, 0, 0] = d_sqrt = D**0.5
    
    # 2. 构造一堆微小的增量 B (0.0004)，在 BF16 下 A+B=A
    v = torch.full((1, L, 1, D), 0.0004, device=device, dtype=dtype)

    # 执行 FA2
    out_fa2 = flash_attn_func(q, k, v, causal=False)
    # 执行 SDPA (Math)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_sdpa = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), is_causal=False
        ).transpose(1,2)

    print(f"    FA2  输出第一个维度均值: {out_fa2[0, 0, 0, :].mean().item():.10f}")
    print(f"    SDPA 输出第一个维度均值: {out_sdpa[0, 0, 0, :].mean().item():.10f}")
    print(f"    💡 如果数值 > 0.0004，说明算子看穿了 BF16 的截断，使用了 FP32 累加。")

if __name__ == "__main__":
    test_accumulator()