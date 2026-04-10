import torch
from flash_attn import flash_attn_func

def verify_precision():
    # 模拟真实长序列
    batch_size, seq_len, nheads, d = 1, 8192, 1, 128
    device = "cuda"
    dtype = torch.bfloat16

    # 构造随机输入
    q = torch.randn((batch_size, seq_len, nheads, d), device=device, dtype=dtype)
    k = torch.randn((batch_size, seq_len, nheads, d), device=device, dtype=dtype)
    v = torch.randn((batch_size, seq_len, nheads, d), device=device, dtype=dtype)

    # --- 1. 获取 FA2 的输出 ---
    # FA2 在底层即便接收 BF16，也会在寄存器中使用 FP32 累加
    out_fa2 = flash_attn_func(q, k, v, causal=True)

    # --- 2. 获取 FP32 参考值 (真理标准) ---
    # 我们把所有数据转为 FP32，模拟最高精度的理想计算
    q_32 = q.permute(0, 2, 1, 3).float() # [B, H, L, D]
    k_32 = k.permute(0, 2, 1, 3).float()
    v_32 = v.permute(0, 2, 1, 3).float()
    
    # 手动算矩阵乘法 [B, H, L, L]
    scores = torch.matmul(q_32, k_32.transpose(-1, -2)) / (d ** 0.5)
    
    # 应用 Causal Mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    probs = torch.softmax(scores, dim=-1)
    out_ref_32 = torch.matmul(probs, v_32)
    # 转回 BF16 用于对比，但它是基于 FP32 计算链生成的
    out_ref_32 = out_ref_32.permute(0, 2, 1, 3).to(dtype) 

    # --- 3. 获取原生 SDPA (BF16) 的输出 ---
    # 强制不使用任何内核优化，走最基础的 BF16 逻辑
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        out_sdpa_bf16 = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), 
            k.permute(0, 2, 1, 3), 
            v.permute(0, 2, 1, 3), 
            is_causal=True
        ).permute(0, 2, 1, 3)

    # --- 4. 结果核对 ---
    diff_fa2 = (out_fa2 - out_ref_32).abs().max().item()
    diff_sdpa = (out_sdpa_bf16 - out_ref_32).abs().max().item()

    print("="*60)
    print(f"🔬 [精度鲁棒性测试结果]")
    print(f"   序列长度: {seq_len} | 数据精度: {dtype}")
    print(f"   FA2  vs. FP32-Ref 最大偏差: {diff_fa2:.10f}")
    print(f"   SDPA vs. FP32-Ref 最大偏差: {diff_sdpa:.10f}")
    print("="*60)

    if diff_fa2 < diff_sdpa:
        ratio = diff_sdpa / diff_fa2 if diff_fa2 > 0 else float('inf')
        print(f"✅ 结论: FA2 的精度是 SDPA 的 {ratio:.2f} 倍！")
        print("   这证明了 FA2 在底层强制使用了 FP32 累加器来抑制误差。")

if __name__ == "__main__":
    verify_precision()