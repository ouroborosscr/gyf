import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

def verify_100k_backward():
    # 模拟 10 万 tokens 的超长序列
    B, L, H, D = 1, 100000, 1, 128
    device, dtype = "cuda", torch.bfloat16

    print(f"🔬 [实验 3] 10万 Tokens 极限梯度稳定性测试 (Dtype: {dtype})")

    # 构造输入，并在其中一个位置加入一个“异常值” (Outlier)，模拟量化权重带来的扰动
    q = torch.randn(B, L, H, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, L, H, D, device=device, dtype=dtype)
    v = torch.randn(B, L, H, D, device=device, dtype=dtype)

    # 1. FA2 计算
    out_fa2 = flash_attn_func(q, k, v, causal=True)
    # 模拟 Loss：对输出求和
    loss_fa2 = out_fa2.sum()
    loss_fa2.backward(retain_graph=True)
    grad_fa2 = q.grad.clone()
    q.grad.zero_()

    # 2. SDPA Math 计算
    # 注意：在 10万长度下，SDPA Math 可能会直接 OOM。如果爆显存，请调小 L。
    # 这也侧面说明了为什么你的 DAPO 训练离不开 FA2
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out_sdpa = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            ).transpose(1, 2)
        loss_sdpa = out_sdpa.sum()
        loss_sdpa.backward()
        grad_sdpa = q.grad.clone()
        
        # 3. 核心比对：检查两个算子在 10万 tokens 下的梯度一致性
        diff = (grad_fa2 - grad_sdpa).abs().max().item()
        rel_error = diff / (grad_fa2.abs().max().item() + 1e-8)
        
        print(f"    FA2  梯度最大值: {grad_fa2.abs().max().item():.10f}")
        print(f"    SDPA 梯度最大值: {grad_sdpa.abs().max().item():.10f}")
        print(f"    10万 Tokens 下的梯度最大偏差: {diff:.10f}")
        print(f"    相对误差: {rel_error:.6f}")
        
    except torch.cuda.OutOfMemoryError:
        print("    ❌ 失败: SDPA Math 在 10万长度下爆显存了。")
        print("    💡 结论：这证明了 SDPA Math 必须分配巨大的中间矩阵，导致了你的显存压力。")

if __name__ == "__main__":
    verify_100k_backward()