"""
Patch transformers 内部的 _flash_attention_forward
用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python intercept_transformers_flash.py
"""
import torch, gc, sys

# ============================================================
# Patch transformers 的 _flash_attention_forward
# ============================================================
import transformers.modeling_flash_attention_utils as flash_utils

_original_flash_forward = flash_utils._flash_attention_forward

def _patched_flash_forward(*args, **kwargs):
    print(f"\n{'='*60}")
    print(f"📌 transformers _flash_attention_forward 被调用!")
    print(f"{'='*60}")
    
    # 打印所有 positional args
    for i, arg in enumerate(args):
        if hasattr(arg, 'shape'):
            print(f"  arg[{i}]: shape={arg.shape}, dtype={arg.dtype}")
        else:
            print(f"  arg[{i}]: {type(arg).__name__} = {arg}")
    
    # 打印所有 kwargs
    for k, v in kwargs.items():
        if hasattr(v, 'shape'):
            nan = torch.isnan(v).any().item() if v.dtype.is_floating_point else False
            print(f"  kwarg {k}: shape={v.shape}, dtype={v.dtype}, nan={nan}")
        else:
            print(f"  kwarg {k}: {v}")
    
    # 检查 q/k/v 的值范围（前 3 个 tensor 参数）
    tensor_args = [a for a in args if hasattr(a, 'shape') and a.dtype.is_floating_point]
    for i, t in enumerate(tensor_args[:3]):
        names = ['query', 'key', 'value']
        name = names[i] if i < len(names) else f'tensor_{i}'
        print(f"  {name}: range=[{t.min().item():.4f}, {t.max().item():.4f}], "
              f"nan={torch.isnan(t).any().item()}, inf={torch.isinf(t).any().item()}")
    
    sys.stdout.flush()
    
    return _original_flash_forward(*args, **kwargs)

flash_utils._flash_attention_forward = _patched_flash_forward

# 同时 patch 导入到其他模块的引用
try:
    import transformers.integrations.flash_attention as fa_integration
    # 这个模块里的 _flash_attention_forward 可能是独立导入的
    if hasattr(fa_integration, '_flash_attention_forward'):
        fa_integration._flash_attention_forward = _patched_flash_forward
except:
    pass

print("✅ transformers flash attention 拦截器已安装\n")

# ============================================================
# 加载模型
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

if hasattr(model.config, "sliding_window"):
    print(f"sliding_window: {model.config.sliding_window}")
    model.config.sliding_window = None

model.eval()

# ============================================================
# forward
# ============================================================
print("\n" + "=" * 70)
print("执行 model forward (seq=256)...")
print("=" * 70)

input_ids = torch.randint(100, 5000, (1, 256), device=device)
try:
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()
    print(f"\n✅ 成功!")
except Exception as e:
    print(f"\n❌ 崩溃: {str(e).split(chr(10))[0][:100]}")

print("\n完成")