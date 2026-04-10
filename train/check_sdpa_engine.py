import torch
import torch.nn.functional as F

def check_engine(enable_math, enable_flash, enable_mem_efficient):
    print(f"\n{'='*60}")
    print(f"enable_math={enable_math}, enable_flash={enable_flash}, enable_mem_efficient={enable_mem_efficient}")
    print("="*60)

    B, H, L, D = 1, 1, 4096, 64
    q = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn(B, H, L, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    # 模拟你代码里的 Left Padding + Causal Mask
    mask = torch.ones(L, L, dtype=torch.bool, device="cuda").tril()
    pad_len = 2000
    mask[:, :pad_len] = False
    attn_mask = torch.zeros(B, H, L, L, dtype=torch.bfloat16, device="cuda")
    attn_mask.masked_fill_(~mask, float("-inf"))

    with torch.backends.cuda.sdp_kernel(
        enable_math=enable_math,
        enable_flash=enable_flash,
        enable_mem_efficient=enable_mem_efficient,
    ):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # 看实际用的哪个 backend（PyTorch 2.1+）
        try:
            from torch.backends.cuda import SDPBackend
            backend_used = torch._C._get_sdp_used_backend()
            backend_name = {SDPBackend.FLASH_ATTENTION: "FLASH",
                            SDPBackend.MATH: "MATH",
                            SDPBackend.EFFICIENT_ATTENTION: "MEM_EFFICIENT",
                            SDPBackend.ERROR: "ERROR"}.get(backend_used, "UNKNOWN")
            print(f"实际 backend: {backend_name}")
        except:
            print("无法获取实际 backend")

        loss = out.sum()
        loss.backward()

    # 看梯度是否稳定
    grad_max = q.grad.abs().max().item()
    grad_nan = torch.isnan(q.grad).any().item()
    grad_inf = torch.isinf(q.grad).any().item()
    print(f"梯度 max = {grad_max:.2f}, nan={grad_nan}, inf={grad_inf}")

if __name__ == "__main__":
    # 模拟你代码里传了自定义 attention_mask 时 PyTorch 的行为
    check_engine(enable_math=False, enable_flash=False, enable_mem_efficient=False)   # 不允许任何加速 → 强制 MATH
    check_engine(enable_math=True, enable_flash=False, enable_mem_efficient=False)    # 允许 MATH
    check_engine(enable_math=False, enable_flash=True, enable_mem_efficient=False)    # 允许 Flash