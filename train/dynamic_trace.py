import torch
from torch.profiler import profile, record_function, ProfilerActivity

def dynamic_trace_test():
    print("="*60)
    print("🔍 [动态调试] PyTorch C++ 算子派发追踪 (ATen Dispatch)")
    print("="*60)

    device = "cuda"
    dtype = torch.bfloat16
    B, H, L, D = 1, 8, 512, 128

    q = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype, requires_grad=True)

    # 构造自定义 Mask (模拟 Left Padding)
    mask = torch.ones(B, H, L, L, device=device, dtype=torch.bool)
    mask[:, :, :, :100] = False 
    attn_mask = torch.zeros(B, H, L, L, device=device, dtype=dtype)
    attn_mask.masked_fill_(~mask, float('-inf'))

    print("\n▶️ 场景 A：无 Mask，使用默认 SDPA (期待动态调用 Flash/MemEff)")
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_a:
        with record_function("SDPA_NO_MASK"):
            out_a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            out_a.sum().backward()
    
    # 打印前 5 个最耗时的 CUDA 算子
    print(prof_a.key_averages().table(sort_by="cuda_time_total", row_limit=5))


    print("\n▶️ 场景 B：传入 Left Padding Mask (期待动态降级到 Math 分步算子)")
    # 重点清理梯度，防止污染
    q.grad, k.grad, v.grad = None, None, None
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof_b:
        with record_function("SDPA_WITH_MASK"):
            out_b = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            out_b.sum().backward()

    print(prof_b.key_averages().table(sort_by="cuda_time_total", row_limit=5))


    print("\n💡 [分析指南]")
    print("1. 如果场景 A 出现了 `aten::_scaled_dot_product_flash_attention`，说明走了融合优化。")
    print("2. 如果场景 B 出现了 `aten::bmm` (矩阵乘法) 和 `aten::_softmax`，这就**动态实锤**了它被拆解成了普通的 Math 算子！")

if __name__ == "__main__":
    dynamic_trace_test()