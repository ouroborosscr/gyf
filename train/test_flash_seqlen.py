"""
精确定位 flash_attn 在 Qwen3.5 (head_dim=256) 下的序列长度崩溃阈值
直接调用 flash_attn_varlen_func，逐步增加序列长度

用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python test_flash_seqlen.py
"""
import torch
import gc

device = "cuda:0"

# Qwen3.5-9B 的 attention 参数
NUM_HEADS = 16       # num_attention_heads
NUM_KV_HEADS = 4     # num_key_value_heads (GQA)
HEAD_DIM = 256       # head_dim
DTYPE = torch.bfloat16

print("=" * 70)
print("flash_attn_varlen_func 序列长度崩溃阈值测试")
print(f"  heads={NUM_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}, dtype={DTYPE}")
print("=" * 70)

from flash_attn import flash_attn_varlen_func, flash_attn_func

# ============================================================
# Part 1: 用 flash_attn_func（非 varlen）测试不同序列长度
# ============================================================
print("\n[Part 1] flash_attn_func (非 varlen, batch=1)")

for seq_len in [256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 8192]:
    try:
        q = torch.randn(1, seq_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        k = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        v = torch.randn(1, seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        
        out = flash_attn_func(q, k, v, causal=True)
        torch.cuda.synchronize()
        
        print(f"  seq={seq_len:>5}: ✅ output shape={out.shape}")
        del q, k, v, out
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  seq={seq_len:>5}: ❌ {err}")
    
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Part 2: flash_attn_varlen_func 测试（模拟 transformers 的调用方式）
# ============================================================
print("\n[Part 2] flash_attn_varlen_func (varlen)")

for seq_len in [256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 8192]:
    try:
        # 模拟单个序列的 varlen 格式
        q = torch.randn(seq_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        k = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        v = torch.randn(seq_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            causal=True,
        )
        torch.cuda.synchronize()
        
        print(f"  seq={seq_len:>5}: ✅ output shape={out.shape}")
        del q, k, v, out
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  seq={seq_len:>5}: ❌ {err}")
    
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Part 3: 多序列 varlen（模拟 batch=4，不同长度）
# ============================================================
print("\n[Part 3] flash_attn_varlen_func (多序列, 模拟 batch=4 不等长)")

test_batches = [
    ("短序列 4x512",   [512, 512, 512, 512]),
    ("混合 512-2048",   [512, 1024, 1536, 2048]),
    ("混合 1024-4096",  [1024, 2048, 3072, 4096]),
    ("全长 4x4096",     [4096, 4096, 4096, 4096]),
    ("超长 4x5120",     [5120, 5120, 5120, 5120]),
    ("超长混合",        [2048, 3072, 4096, 5120]),
    ("极端混合",        [512, 1024, 4096, 6144]),
]

for label, seq_lens in test_batches:
    try:
        total_len = sum(seq_lens)
        q = torch.randn(total_len, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        k = torch.randn(total_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        v = torch.randn(total_len, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0).numpy()), 
                                   dtype=torch.int32, device=device)
        max_seqlen = max(seq_lens)
        
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        torch.cuda.synchronize()
        
        print(f"  {label:<20} (max={max_seqlen:>5}): ✅")
        del q, k, v, out
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  {label:<20} (max={max_seqlen:>5}): ❌ {err}")
    
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# Part 4: 二分查找精确阈值
# ============================================================
print("\n[Part 4] 二分查找精确崩溃阈值 (varlen, batch=1)")

low, high = 256, 8192
last_ok = low

while low <= high:
    mid = (low + high) // 2
    # 对齐到 128（flash_attn 内部对齐要求）
    mid = (mid // 128) * 128
    if mid == 0:
        mid = 128
    
    try:
        q = torch.randn(mid, NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        k = torch.randn(mid, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        v = torch.randn(mid, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        
        cu_seqlens = torch.tensor([0, mid], dtype=torch.int32, device=device)
        
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen_q=mid,
            max_seqlen_k=mid,
            causal=True,
        )
        torch.cuda.synchronize()
        
        last_ok = mid
        low = mid + 128
        del q, k, v, out
    except:
        high = mid - 128
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n  最大可用序列长度: {last_ok}")
print(f"  (prompt+completion 总长度不能超过这个值)")

# ============================================================
# Part 5: 对比 head_dim=128 vs 256
# ============================================================
print("\n[Part 5] 对比 head_dim=128 vs 256 (seq=4096)")

for hd in [128, 256]:
    try:
        q = torch.randn(4096, NUM_HEADS, hd, dtype=DTYPE, device=device)
        k = torch.randn(4096, NUM_KV_HEADS, hd, dtype=DTYPE, device=device)
        v = torch.randn(4096, NUM_KV_HEADS, hd, dtype=DTYPE, device=device)
        
        cu_seqlens = torch.tensor([0, 4096], dtype=torch.int32, device=device)
        
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens, cu_seqlens,
            max_seqlen_q=4096,
            max_seqlen_k=4096,
            causal=True,
        )
        torch.cuda.synchronize()
        
        print(f"  head_dim={hd}: ✅")
        del q, k, v, out
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  head_dim={hd}: ❌ {err}")
    
    gc.collect()
    torch.cuda.empty_cache()

print(f"\n{'='*70}")
print("测试完成")
print(f"{'='*70}")