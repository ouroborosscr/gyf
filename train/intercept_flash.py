"""
在 flash_attn 崩溃前拦截参数
用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python intercept_flash.py
"""
import torch, gc, sys

# ============================================================
# 在 import transformers 之前先 patch flash_attn
# ============================================================
import flash_attn.flash_attn_interface as fai

_original_varlen = fai.flash_attn_varlen_func
_original_func = fai.flash_attn_func

def _intercept_varlen(*args, **kwargs):
    print(f"\n{'='*50}")
    print(f"📌 flash_attn_varlen_func 被调用!")
    print(f"{'='*50}")
    
    # 解析参数
    q = args[0] if len(args) > 0 else kwargs.get('q')
    k = args[1] if len(args) > 1 else kwargs.get('k')
    v = args[2] if len(args) > 2 else kwargs.get('v')
    cu_q = args[3] if len(args) > 3 else kwargs.get('cu_seqlens_q')
    cu_k = args[4] if len(args) > 4 else kwargs.get('cu_seqlens_k')
    max_sq = args[5] if len(args) > 5 else kwargs.get('max_seqlen_q')
    max_sk = args[6] if len(args) > 6 else kwargs.get('max_seqlen_k')
    
    print(f"  q: shape={q.shape}, dtype={q.dtype}, contiguous={q.is_contiguous()}")
    print(f"  k: shape={k.shape}, dtype={k.dtype}, contiguous={k.is_contiguous()}")
    print(f"  v: shape={v.shape}, dtype={v.dtype}, contiguous={v.is_contiguous()}")
    print(f"  cu_seqlens_q: {cu_q} (dtype={cu_q.dtype})")
    print(f"  cu_seqlens_k: {cu_k} (dtype={cu_k.dtype})")
    print(f"  max_seqlen_q: {max_sq}")
    print(f"  max_seqlen_k: {max_sk}")
    
    # 检查合法性
    if q is not None:
        total_q = q.shape[0]
        expected = cu_q[-1].item() if cu_q is not None else "N/A"
        match = "✅" if total_q == expected else f"❌ MISMATCH"
        print(f"  q total tokens: {total_q}, cu_seqlens_q[-1]: {expected} {match}")
    
    # 打印所有 kwargs
    for kw, val in kwargs.items():
        if kw not in ['q', 'k', 'v', 'cu_seqlens_q', 'cu_seqlens_k', 'max_seqlen_q', 'max_seqlen_k']:
            print(f"  kwarg {kw}: {val}")
    
    # 检查额外的 positional args
    named_count = 7  # q, k, v, cu_q, cu_k, max_sq, max_sk
    if len(args) > named_count:
        for i in range(named_count, len(args)):
            print(f"  extra arg[{i}]: {type(args[i]).__name__} = {args[i]}")
    
    # 检查 NaN/Inf
    if q is not None:
        q_nan = torch.isnan(q).any().item()
        q_inf = torch.isinf(q).any().item()
        print(f"  q has NaN: {q_nan}, Inf: {q_inf}")
    if k is not None:
        k_nan = torch.isnan(k).any().item()
        k_inf = torch.isinf(k).any().item()
        print(f"  k has NaN: {k_nan}, Inf: {k_inf}")
    if v is not None:
        v_nan = torch.isnan(v).any().item()
        v_inf = torch.isinf(v).any().item()
        print(f"  v has NaN: {v_nan}, Inf: {v_inf}")
    
    # 打印调用栈的关键几帧
    import traceback
    stack = traceback.extract_stack()
    print(f"\n  调用栈 (最近 5 帧):")
    for frame in stack[-6:-1]:
        fname = frame.filename.split('/')[-1]
        print(f"    {fname}:{frame.lineno} in {frame.name}")
    
    sys.stdout.flush()
    
    return _original_varlen(*args, **kwargs)


def _intercept_func(*args, **kwargs):
    print(f"\n📌 flash_attn_func 被调用!")
    q = args[0] if len(args) > 0 else kwargs.get('q')
    print(f"  q: shape={q.shape}, dtype={q.dtype}")
    for kw, val in kwargs.items():
        if kw not in ['q', 'k', 'v']:
            print(f"  kwarg {kw}: {val}")
    sys.stdout.flush()
    return _original_func(*args, **kwargs)


# 安装拦截器
fai.flash_attn_varlen_func = _intercept_varlen
fai.flash_attn_func = _intercept_func

# 同时 patch FlashAttnVarlenFunc.apply (transformers 可能直接调用这个)
if hasattr(fai, 'FlashAttnVarlenFunc'):
    _original_apply = fai.FlashAttnVarlenFunc.apply
    # 不能直接 patch .apply，改为 patch forward

print("✅ flash_attn 拦截器已安装\n")

# ============================================================
# 现在加载模型
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)

print(f"sliding_window: {getattr(model.config, 'sliding_window', 'N/A')}")
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None
    print("sliding_window 已设为 None")

model.eval()

# ============================================================
# 做一次 forward
# ============================================================
print("\n" + "=" * 70)
print("执行 model forward (seq=256)...")
print("=" * 70)

input_ids = torch.randint(100, 5000, (1, 256), device=device)
try:
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()
    print(f"\n✅ 成功! logits shape={out.logits.shape}")
except Exception as e:
    torch.cuda.synchronize()
    print(f"\n❌ 崩溃: {str(e).split(chr(10))[0][:100]}")

print("\n完成")