"""
确认重编译 + 精确复现 transformers 调用路径
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_final.py
"""
import torch
import os, subprocess, sys

device = "cuda:0"

# ============================================================
print("=" * 60)
print("Step 0: 确认 flash_attn 是否真的重编译了")
print("=" * 60)

import flash_attn
fa_dir = os.path.dirname(flash_attn.__file__)
print(f"flash_attn 版本: {flash_attn.__version__}")
print(f"flash_attn 目录: {fa_dir}")

# 查找 .so 文件和修改时间
for root, dirs, files in os.walk(os.path.dirname(fa_dir)):
    for f in files:
        if "flash_attn" in f and f.endswith(".so"):
            full = os.path.join(root, f)
            mtime = os.path.getmtime(full)
            from datetime import datetime
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            size_mb = os.path.getsize(full) / 1024 / 1024
            print(f"  {f}: 修改时间={mtime_str}, 大小={size_mb:.1f}MB")

# 检查编译时的 CUDA 架构
print(f"\n当前 GPU: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"SM 架构: {props.major}.{props.minor}")
print(f"PyTorch CUDA: {torch.version.cuda}")

# ============================================================
print("\n" + "=" * 60)
print("Step 1: 裸测 flash_attn (Qwen3.5-9B 实际参数)")
print("=" * 60)

from flash_attn import flash_attn_varlen_func

# Qwen3.5-9B 实际参数: 16 heads, 4 kv_heads, head_dim=256
num_heads = 16
num_kv_heads = 4
head_dim = 256

for seq_len in [64, 256, 1024]:
    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    
    # 用和 transformers 一样的 softmax_scale
    softmax_scale = 0.0625  # 1/sqrt(256) = 1/16
    
    try:
        out = flash_attn_varlen_func(
            q, k, v, cu, cu, seq_len, seq_len,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
        )
        torch.cuda.synchronize()
        print(f"  裸调用 seq_len={seq_len:>5}: ✅ 通过")
    except Exception as e:
        print(f"  裸调用 seq_len={seq_len:>5}: ❌ 失败 - {e}")

# ============================================================
print("\n" + "=" * 60)
print("Step 2: 通过 transformers 的 _flash_attention_forward 调用")
print("=" * 60)

try:
    from transformers.modeling_flash_attention_utils import _flash_attention_forward
    
    seq_len = 64
    batch_size = 1
    
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)

    try:
        out = _flash_attention_forward(
            query, key, value,
            None,  # attention_mask
            query_length=seq_len,
            is_causal=True,
            dropout=0.0,
            softmax_scale=softmax_scale,
            sliding_window=None,
            softcap=None,
            use_top_left_mask=False,
            target_dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()
        print(f"  transformers _flash_attention_forward: ✅ 通过")
    except Exception as e:
        print(f"  transformers _flash_attention_forward: ❌ 失败 - {e}")
        import traceback; traceback.print_exc()
except Exception as e:
    print(f"  无法导入 _flash_attention_forward: {e}")

# ============================================================
print("\n" + "=" * 60)
print("Step 3: 加载模型，用 SDPA 而不是 flash_attn (对照组)")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model_sdpa = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

input_ids = torch.randint(0, 1000, (1, 256), device=device)
try:
    with torch.no_grad():
        out = model_sdpa(input_ids=input_ids)
    torch.cuda.synchronize()
    print(f"  SDPA forward: ✅ 通过 (logits shape={out.logits.shape})")
    sdpa_ok = True
except Exception as e:
    print(f"  SDPA forward: ❌ 失败 - {e}")
    sdpa_ok = False

del model_sdpa
import gc; gc.collect()
torch.cuda.empty_cache()

# ============================================================
print("\n" + "=" * 60)
print("Step 4: 加载模型，用 flash_attention_2")
print("=" * 60)

try:
    torch.cuda.synchronize()
except:
    print("  CUDA 已损坏，跳过")
    sys.exit(1)

model_fa = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

if hasattr(model_fa.config, "sliding_window"):
    model_fa.config.sliding_window = None

input_ids = torch.randint(0, 1000, (1, 64), device=device)
try:
    with torch.no_grad():
        out = model_fa(input_ids=input_ids)
    torch.cuda.synchronize()
    print(f"  flash_attention_2 forward: ✅ 通过")
except Exception as e:
    print(f"  flash_attention_2 forward: ❌ 失败 - {e}")
    
    # 查看 transformers 具体怎么调用的
    print(f"\n  --- 检查 transformers flash_attention 集成代码 ---")
    import transformers.modeling_flash_attention_utils as tfau
    import inspect
    src = inspect.getsource(tfau._flash_attention_forward)
    # 只打印关键的调用部分
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if "flash_varlen" in line or "flash_attn" in line or "varlen_fn" in line:
            start = max(0, i-2)
            end = min(len(lines), i+3)
            print(f"\n  行 {start}-{end}:")
            for j in range(start, end):
                print(f"    {j}: {lines[j]}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
if sdpa_ok:
    print("SDPA 可用。如果 flash_attention_2 仍然失败，建议训练时用 SDPA：")
    print("  在 train_grpo.py 中去掉 --use_flash_attn 参数")
    print("  或者在代码中改为 attn_implementation='sdpa'")
    print("  PyTorch 2.10 的 SDPA 对 A100 有很好的优化，性能差距不大。")