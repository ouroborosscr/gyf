"""
逐步加变量：定位是 4bit / prepare_kbit / LoRA / gc 哪个组合导致崩溃
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_model_layers.py
"""
import torch
import gc

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

def test_config(label, use_4bit, use_prepare_kbit, use_lora, use_gc, use_flash):
    """加载模型并做一次 forward+backward"""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  4bit={use_4bit} prepare_kbit={use_prepare_kbit} lora={use_lora} gc={use_gc} flash={use_flash}")
    print(f"{'='*60}")
    
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "attn_implementation": "flash_attention_2" if use_flash else "sdpa",
        "low_cpu_mem_usage": True,
    }
    if use_4bit:
        kwargs["quantization_config"] = bnb_config
    
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **kwargs)
        if hasattr(model.config, "sliding_window"):
            model.config.sliding_window = None
        
        if use_prepare_kbit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gc)
        
        if use_lora:
            model.enable_input_require_grads()
            model = get_peft_model(model, peft_config)
        elif not use_prepare_kbit:
            model.enable_input_require_grads()
        
        model.train()
        if use_gc and not use_prepare_kbit:
            model.gradient_checkpointing_enable()
        
        input_ids = torch.randint(0, 1000, (1, 256), device=device)
        labels = input_ids.clone()
        
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_val = loss.item()
        loss.backward()
        torch.cuda.synchronize()
        
        # 检查梯度
        grad_nans = 0
        grad_total = 0
        for n, p in model.named_parameters():
            if p.grad is not None:
                grad_total += 1
                if torch.isnan(p.grad).any():
                    grad_nans += 1
        
        if grad_nans == 0:
            print(f"  ✅ loss={loss_val:.4f}, grad_nan=0/{grad_total}")
        else:
            print(f"  ❌ loss={loss_val:.4f}, grad_nan={grad_nans}/{grad_total}")
        
        result = "pass"
    except Exception as e:
        err = str(e).split('\n')[0]
        print(f"  ❌ 崩溃: {err}")
        result = "crash"
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 检查 CUDA 是否还活着
    try:
        torch.cuda.synchronize()
        return result
    except:
        print("  ⚠️ CUDA 上下文已损坏，后续测试不可靠")
        return "cuda_dead"

# ============================================================
# 逐步加变量
# ============================================================

# A: bf16 + flash_attn + backward (不量化，不 LoRA)
r = test_config("A: bf16 + flash + backward (基线)", 
                use_4bit=False, use_prepare_kbit=False, use_lora=False, use_gc=False, use_flash=True)
if r == "cuda_dead": exit()

# B: bf16 + flash + gradient_checkpointing
r = test_config("B: bf16 + flash + gc",
                use_4bit=False, use_prepare_kbit=False, use_lora=False, use_gc=True, use_flash=True)
if r == "cuda_dead": exit()

# C: 4bit + flash (不 prepare_kbit, 不 gc)
r = test_config("C: 4bit + flash (裸)",
                use_4bit=True, use_prepare_kbit=False, use_lora=False, use_gc=False, use_flash=True)
if r == "cuda_dead": exit()

# D: 4bit + flash + prepare_kbit (这步会把 LN 转 fp32)
r = test_config("D: 4bit + flash + prepare_kbit",
                use_4bit=True, use_prepare_kbit=True, use_lora=False, use_gc=False, use_flash=True)
if r == "cuda_dead": exit()

# E: 4bit + flash + prepare_kbit + gc
r = test_config("E: 4bit + flash + prepare_kbit + gc",
                use_4bit=True, use_prepare_kbit=True, use_lora=False, use_gc=True, use_flash=True)
if r == "cuda_dead": exit()

# F: 4bit + flash + prepare_kbit + gc + LoRA (完整配置)
r = test_config("F: 4bit + flash + prepare_kbit + gc + LoRA (你的完整配置)",
                use_4bit=True, use_prepare_kbit=True, use_lora=True, use_gc=True, use_flash=True)
if r == "cuda_dead": exit()

# G: 对照 - 把 F 的 flash 换成 sdpa
r = test_config("G: 4bit + SDPA + prepare_kbit + gc + LoRA (对照组)",
                use_4bit=True, use_prepare_kbit=True, use_lora=True, use_gc=True, use_flash=False)

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)