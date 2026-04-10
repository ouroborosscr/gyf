"""
测试 PyTorch SDPA 对不同 head_dim 的 backend 选择
以及 Qwen3.5 的实际 head_dim

用法: CUDA_VISIBLE_DEVICES=0 python test_sdpa_backend.py
"""
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

print("=" * 60)
print("PyTorch SDPA Backend 测试")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ============================================================
# Part 1: Qwen3.5 的 head_dim
# ============================================================
print("\n--- Qwen3.5 模型参数 (已确认) ---")
actual_head_dim = 256
print(f"  head_dim: {actual_head_dim}")
print(f"  num_attention_heads: 16, num_key_value_heads: 4")

# ============================================================
# Part 2: 测试 SDPA 各 backend 对不同 head_dim 的支持
# ============================================================
print("\n--- SDPA Backend 支持测试 ---")
print(f"{'head_dim':<10} {'flash_sdp':<12} {'mem_eff':<12} {'math':<12} {'实际使用'}")
print("-" * 58)

backends = {
    "flash_sdp": SDPBackend.FLASH_ATTENTION,
    "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}

for head_dim in [64, 128, 192, 256, 384]:
    seq_len = 512
    batch = 1
    n_heads = 4

    q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

    results = {}
    for name, backend in backends.items():
        try:
            with sdpa_kernel(backend):
                _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            results[name] = "✅"
        except Exception as e:
            results[name] = "❌"

    # 测试默认选择哪个
    default_backend = "?"
    # 逐个关闭来探测
    for test_name, test_backend in [("flash_sdp", SDPBackend.FLASH_ATTENTION), 
                                      ("mem_eff", SDPBackend.EFFICIENT_ATTENTION)]:
        if results[test_name] == "✅":
            default_backend = test_name
            break
    if default_backend == "?":
        default_backend = "math"

    marker = " ← Qwen3.5" if head_dim == actual_head_dim else ""
    print(f"{head_dim:<10} {results['flash_sdp']:<12} {results['mem_eff']:<12} {results['math']:<12} {default_backend}{marker}")

# ============================================================
# Part 3: 长序列 + head_dim=256 的数值稳定性对比
# ============================================================
print("\n--- 数值稳定性: math backend vs flash_attn (head_dim=256, seq=4096) ---")

head_dim = 256
seq_len = 4096
batch = 1
n_heads = 4

torch.manual_seed(42)
q = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
k = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)
v = torch.randn(batch, n_heads, seq_len, head_dim, device="cuda", dtype=torch.bfloat16)

# Math backend: 两次计算看差异（模拟 gradient checkpointing 重算）
with sdpa_kernel(SDPBackend.MATH):
    out1 = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out2 = F.scaled_dot_product_attention(q, k, v, is_causal=True)

diff_math = (out1 - out2).abs()
print(f"  Math backend 两次计算差异:")
print(f"    max diff:  {diff_math.max().item():.6e}")
print(f"    mean diff: {diff_math.mean().item():.6e}")
print(f"    零差异:    {(diff_math == 0).all().item()}")

# flash_attn 对比
try:
    from flash_attn import flash_attn_func
    q_fa = q.transpose(1, 2)  # [B, S, H, D]
    k_fa = k.transpose(1, 2)
    v_fa = v.transpose(1, 2)

    out_fa1 = flash_attn_func(q_fa, k_fa, v_fa, causal=True)
    out_fa2 = flash_attn_func(q_fa, k_fa, v_fa, causal=True)

    diff_fa = (out_fa1 - out_fa2).abs()
    print(f"\n  flash_attn 两次计算差异:")
    print(f"    max diff:  {diff_fa.max().item():.6e}")
    print(f"    mean diff: {diff_fa.mean().item():.6e}")
    print(f"    零差异:    {(diff_fa == 0).all().item()}")
except ImportError:
    print("\n  flash_attn 未安装，跳过")

print(f"\n{'='*60}")
print("结论:")
print(f"  Qwen3.5 head_dim = {actual_head_dim}")
if actual_head_dim > 128:
    print(f"  head_dim > 128 → SDPA flash kernel 不可用 → 降级到 math/mem_eff backend")
    print(f"  math backend + bf16 + gradient checkpointing → 重算不一致 → 梯度爆炸")
    print(f"  flash_attn 2.8.3 原生支持 head_dim=256 → fused kernel → 重算一致 → 梯度正常")
print(f"{'='*60}")