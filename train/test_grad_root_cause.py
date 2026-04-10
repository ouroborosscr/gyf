"""
测试梯度爆炸的真正原因
在两个环境分别跑：
  scr_train3 (Python 3.12, 未修复 position_ids bug): 不加 flash_attn
  scr_train2 (Python 3.11, 已修复 position_ids bug): 加 flash_attn

用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python test_grad_root_cause.py [--flash_attn]
"""
import torch, gc, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--flash_attn", action="store_true", help="使用 flash_attention_2")
args = parser.parse_args()

device = "cuda:0"
MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

attn_impl = "flash_attention_2" if args.flash_attn else "sdpa"

print("=" * 70)
print(f"测试配置:")
print(f"  Python: {sys.version.split()[0]}")
print(f"  PyTorch: {torch.__version__}")
print(f"  Attention: {attn_impl}")
print(f"  设备: {device}")

try:
    import flash_attn
    print(f"  flash_attn: {flash_attn.__version__}")
except:
    print(f"  flash_attn: 未安装")

# 检查 position_ids bug 是否已修复
import transformers.modeling_flash_attention_utils as fau
import inspect
src = inspect.getsource(fau._is_packed_sequence)
patched = "position_ids.dim() > 2" in src
print(f"  position_ids bug 修复: {'✅ 已修复' if patched else '❌ 未修复'}")
print("=" * 70)

# 加载模型
print("\n加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    attn_implementation=attn_impl, low_cpu_mem_usage=True,
)
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.train()

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=5e-7
)

# GRPO 风格的 forward + backward
def test_grpo_gradient(seq_len):
    optimizer.zero_grad()
    input_ids = torch.randint(100, 5000, (1, seq_len), device=device)
    
    try:
        out = model(input_ids=input_ids, use_cache=False)
    except Exception as e:
        return None, None, str(e)[:80]
    
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    per_token_logps = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    
    advantage = torch.tensor([1.5], device=device, dtype=torch.bfloat16)
    per_token_loss = -advantage * per_token_logps
    mask = torch.ones_like(per_token_logps)
    loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).mean()
    
    loss.backward()
    torch.cuda.synchronize()
    
    # 收集梯度
    grad_max = 0.0
    grad_nan = 0
    worst_param = ""
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                grad_nan += 1
            else:
                gm = p.grad.abs().max().item()
                if gm > grad_max:
                    grad_max = gm
                    worst_param = name
    
    optimizer.zero_grad()
    del out, logits, log_probs, per_token_logps, per_token_loss, loss, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    
    return grad_max, grad_nan, worst_param

# 跑测试
print("\n" + "=" * 70)
print(f"GRPO 梯度测试 ({attn_impl})")
print("=" * 70)

for seq_len in [256, 512, 1024, 2048]:
    grad_max, grad_nan, info = test_grpo_gradient(seq_len)
    
    if grad_max is None:
        print(f"  seq={seq_len:>5}: ❌ {info}")
        break
    else:
        gmax_str = f"{grad_max:.2e}"
        status = "✅" if grad_max < 1.0 else ("⚠️" if grad_max < 1e10 else "❌ 爆炸")
        print(f"  seq={seq_len:>5}: {status} grad_max={gmax_str} nan={grad_nan} @ {info.split('.')[-1][:30]}")

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)