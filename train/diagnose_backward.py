"""
精确测试 flash_attn backward 是否产生 NaN
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_backward.py
"""
import torch
import torch.nn.functional as F

device = "cuda:0"
dtype = torch.bfloat16

# Qwen3.5-9B 参数
num_heads = 16
num_kv_heads = 4
head_dim = 256
softmax_scale = 1.0 / (head_dim ** 0.5)  # 0.0625

print("=" * 60)
print("环境")
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA:    {torch.version.cuda}")
import flash_attn
print(f"flash_attn: {flash_attn.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"head_dim={head_dim}, softmax_scale={softmax_scale}")

from flash_attn import flash_attn_varlen_func, flash_attn_func

# ============================================================
print("\n" + "=" * 60)
print("Test 1: flash_attn_func forward + backward (非 varlen)")
print("=" * 60)

for seq_len in [64, 256, 1024, 2048]:
    torch.cuda.empty_cache()
    
    q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    try:
        out = flash_attn_func(q, k, v, causal=True, softmax_scale=softmax_scale)
        torch.cuda.synchronize()
        
        fwd_nan = torch.isnan(out).any().item()
        
        # backward
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        q_grad_nan = torch.isnan(q.grad).any().item()
        k_grad_nan = torch.isnan(k.grad).any().item()
        v_grad_nan = torch.isnan(v.grad).any().item()
        
        status = "✅" if not (fwd_nan or q_grad_nan or k_grad_nan or v_grad_nan) else "❌"
        print(f"  seq={seq_len:>5}: {status} fwd_nan={fwd_nan}, "
              f"q_grad_nan={q_grad_nan}, k_grad_nan={k_grad_nan}, v_grad_nan={v_grad_nan}")
    except Exception as e:
        print(f"  seq={seq_len:>5}: ❌ 异常 - {e}")

# ============================================================
print("\n" + "=" * 60)
print("Test 2: flash_attn_varlen_func forward + backward (transformers 用的路径)")
print("=" * 60)

for seq_len in [64, 256, 1024, 2048]:
    torch.cuda.empty_cache()
    
    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    
    try:
        out = flash_attn_varlen_func(q, k, v, cu, cu, seq_len, seq_len,
                                      causal=True, softmax_scale=softmax_scale)
        torch.cuda.synchronize()
        fwd_nan = torch.isnan(out).any().item()
        
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        q_grad_nan = torch.isnan(q.grad).any().item()
        k_grad_nan = torch.isnan(k.grad).any().item()
        v_grad_nan = torch.isnan(v.grad).any().item()
        
        status = "✅" if not (fwd_nan or q_grad_nan or k_grad_nan or v_grad_nan) else "❌"
        print(f"  seq={seq_len:>5}: {status} fwd_nan={fwd_nan}, "
              f"q_grad_nan={q_grad_nan}, k_grad_nan={k_grad_nan}, v_grad_nan={v_grad_nan}")
    except Exception as e:
        print(f"  seq={seq_len:>5}: ❌ 异常 - {e}")

# ============================================================
print("\n" + "=" * 60)
print("Test 3: 对比 head_dim=128 vs 256 的 backward")
print("=" * 60)

for hd in [64, 128, 192, 256]:
    torch.cuda.empty_cache()
    seq_len = 512
    scale = 1.0 / (hd ** 0.5)
    
    q = torch.randn(1, seq_len, num_heads, hd, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, seq_len, num_kv_heads, hd, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, seq_len, num_kv_heads, hd, device=device, dtype=dtype, requires_grad=True)
    
    try:
        out = flash_attn_func(q, k, v, causal=True, softmax_scale=scale)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        any_nan = (torch.isnan(q.grad).any().item() or 
                   torch.isnan(k.grad).any().item() or 
                   torch.isnan(v.grad).any().item())
        
        grad_max = max(q.grad.abs().max().item(), k.grad.abs().max().item(), v.grad.abs().max().item())
        
        status = "✅" if not any_nan else "❌"
        print(f"  head_dim={hd:>3} (scale={scale:.4f}): {status} any_nan={any_nan}, grad_max={grad_max:.2f}")
    except Exception as e:
        print(f"  head_dim={hd:>3}: ❌ 异常 - {e}")

# ============================================================
print("\n" + "=" * 60)
print("Test 4: gradient checkpointing 模拟 (torch.utils.checkpoint)")
print("=" * 60)

from torch.utils.checkpoint import checkpoint

def flash_attn_step(q, k, v):
    """模拟 transformers 在 gradient_checkpointing 下的调用"""
    return flash_attn_func(q, k, v, causal=True, softmax_scale=softmax_scale)

for seq_len in [256, 1024, 2048]:
    torch.cuda.empty_cache()
    
    q = torch.randn(1, seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    try:
        # 用 checkpoint 包裹（和训练时一样，backward 时会重算 forward）
        out = checkpoint(flash_attn_step, q, k, v, use_reentrant=False)
        torch.cuda.synchronize()
        fwd_nan = torch.isnan(out).any().item()
        
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        any_nan = (torch.isnan(q.grad).any().item() or 
                   torch.isnan(k.grad).any().item() or 
                   torch.isnan(v.grad).any().item())
        
        status = "✅" if not (fwd_nan or any_nan) else "❌"
        print(f"  seq={seq_len:>5} + checkpoint: {status} fwd_nan={fwd_nan}, grad_nan={any_nan}")
    except Exception as e:
        print(f"  seq={seq_len:>5} + checkpoint: ❌ 异常 - {e}")

# ============================================================
print("\n" + "=" * 60)
print("Test 5: 通过真实模型做 1 步训练 (forward + backward)")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
peft_model = get_peft_model(model, peft_config)
peft_model.train()
peft_model.gradient_checkpointing_enable()

print("模型加载完成，测试 forward + backward...")

for seq_len in [256, 512, 1024, 2048]:
    torch.cuda.empty_cache()
    peft_model.zero_grad()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    labels = input_ids.clone()
    
    try:
        out = peft_model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_val = loss.item()
        loss_nan = torch.isnan(loss).item()
        
        loss.backward()
        torch.cuda.synchronize()
        
        # 检查所有 LoRA 参数的梯度
        grad_nans = 0
        grad_total = 0
        grad_max = 0.0
        for name, p in peft_model.named_parameters():
            if p.grad is not None:
                grad_total += 1
                if torch.isnan(p.grad).any():
                    grad_nans += 1
                gmax = p.grad.abs().max().item()
                if gmax > grad_max:
                    grad_max = gmax
        
        status = "✅" if (grad_nans == 0 and not loss_nan) else "❌"
        print(f"  seq={seq_len:>5}: {status} loss={loss_val:.4f} loss_nan={loss_nan}, "
              f"grad_nan_params={grad_nans}/{grad_total}, grad_max={grad_max:.2f}")
    except Exception as e:
        print(f"  seq={seq_len:>5}: ❌ 异常 - {e}")
    
    peft_model.zero_grad()

# ============================================================
print("\n" + "=" * 60)
print("Test 6: 同样配置换 SDPA (对照组)")
print("=" * 60)

del peft_model, model
import gc; gc.collect(); torch.cuda.empty_cache()

model2 = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)
if hasattr(model2.config, "sliding_window"):
    model2.config.sliding_window = None

model2 = prepare_model_for_kbit_training(model2, use_gradient_checkpointing=True)
model2.enable_input_require_grads()
peft_model2 = get_peft_model(model2, peft_config)
peft_model2.train()
peft_model2.gradient_checkpointing_enable()

print("SDPA 模型加载完成...")

for seq_len in [256, 512, 1024, 2048]:
    torch.cuda.empty_cache()
    peft_model2.zero_grad()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    labels = input_ids.clone()
    
    try:
        out = peft_model2(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_val = loss.item()
        loss_nan = torch.isnan(loss).item()
        
        loss.backward()
        torch.cuda.synchronize()
        
        grad_nans = 0
        grad_total = 0
        grad_max = 0.0
        for name, p in peft_model2.named_parameters():
            if p.grad is not None:
                grad_total += 1
                if torch.isnan(p.grad).any():
                    grad_nans += 1
                gmax = p.grad.abs().max().item()
                if gmax > grad_max:
                    grad_max = gmax
        
        status = "✅" if (grad_nans == 0 and not loss_nan) else "❌"
        print(f"  seq={seq_len:>5}: {status} loss={loss_val:.4f} loss_nan={loss_nan}, "
              f"grad_nan_params={grad_nans}/{grad_total}, grad_max={grad_max:.2f}")
    except Exception as e:
        print(f"  seq={seq_len:>5}: ❌ 异常 - {e}")
    
    peft_model2.zero_grad()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)