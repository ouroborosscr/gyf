"""
复现：2 步训练后模型是否产生 NaN logits
逐步检查每一步训练后的权重、logits、梯度状态
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_2step_nan.py
"""
import torch
import gc
import json
import subprocess, sys

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

WORKER = '''
import torch, json, sys, gc, os

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

cfg = json.loads(sys.argv[1])
use_deepspeed = cfg["use_deepspeed"]
use_detect_anomaly = cfg["use_detect_anomaly"]
use_prepare_kbit = cfg["use_prepare_kbit"]
num_steps = cfg["num_steps"]
seq_len = cfg["seq_len"]

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None

model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
if hasattr(model, "generation_config"):
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

if use_prepare_kbit:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model.enable_input_require_grads()
model = get_peft_model(model, peft_config)
model.train()

if use_detect_anomaly:
    torch.autograd.set_detect_anomaly(True)

# 用 adamw_torch，和训练配置一致
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=5e-7
)

def check_model_health(label):
    """检查模型权重和 logits 是否健康"""
    # 检查 LoRA 参数
    lora_nan = 0
    lora_inf = 0
    lora_total = 0
    lora_max = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and p.data is not None:
            lora_total += 1
            if torch.isnan(p.data).any():
                lora_nan += 1
            if torch.isinf(p.data).any():
                lora_inf += 1
            m = p.data.abs().max().item()
            if m > lora_max:
                lora_max = m

    # 做一次 forward 检查 logits
    test_ids = torch.randint(0, 1000, (1, 64), device=device)
    with torch.no_grad():
        out = model(input_ids=test_ids)
    logits = out.logits[0, -1, :]
    logits_nan = torch.isnan(logits).any().item()
    logits_inf = torch.isinf(logits).any().item()
    logits_max = logits.abs().max().item() if not (logits_nan or logits_inf) else float('inf')
    
    # 检查 softmax
    probs = torch.softmax(logits.float(), dim=-1)
    probs_ok = not (torch.isnan(probs).any().item() or (probs < 0).any().item())
    
    result = {
        "label": label,
        "lora_nan": lora_nan,
        "lora_inf": lora_inf,
        "lora_total": lora_total,
        "lora_max": round(lora_max, 6),
        "logits_nan": logits_nan,
        "logits_inf": logits_inf,
        "logits_max": round(logits_max, 2) if logits_max < 1e10 else "inf",
        "probs_ok": probs_ok,
    }
    
    healthy = (lora_nan == 0 and lora_inf == 0 and not logits_nan and not logits_inf and probs_ok)
    return result, healthy

# 初始检查
result, healthy = check_model_health("初始状态")
print("CHECK:" + json.dumps(result))

# 训练循环
for step in range(1, num_steps + 1):
    optimizer.zero_grad()
    
    input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
    labels = input_ids.clone()
    
    try:
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss_val = loss.item()
        loss_nan = torch.isnan(loss).item()
        
        loss.backward()
        torch.cuda.synchronize()
        
        # 检查梯度
        grad_nan_count = 0
        grad_inf_count = 0
        grad_max = 0.0
        grad_total = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad_total += 1
                if torch.isnan(p.grad).any():
                    grad_nan_count += 1
                if torch.isinf(p.grad).any():
                    grad_inf_count += 1
                gm = p.grad.abs().max().item()
                if gm > grad_max:
                    grad_max = gm
        
        # 手动梯度裁剪（和 Trainer 一样）
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=0.3
        )
        
        # 检查裁剪后的梯度
        grad_max_after = 0.0
        for p in model.parameters():
            if p.grad is not None:
                gm = p.grad.abs().max().item()
                if gm > grad_max_after:
                    grad_max_after = gm
        
        optimizer.step()
        torch.cuda.synchronize()
        
        step_info = {
            "step": step,
            "loss": round(loss_val, 4),
            "loss_nan": loss_nan,
            "grad_nan": grad_nan_count,
            "grad_inf": grad_inf_count,
            "grad_max_before_clip": round(grad_max, 4),
            "grad_max_after_clip": round(grad_max_after, 6),
            "grad_total": grad_total,
        }
        print("STEP:" + json.dumps(step_info))
        
        # 训练后检查健康
        result, healthy = check_model_health(f"Step {step} 后")
        print("CHECK:" + json.dumps(result))
        
        if not healthy:
            print("DEAD:模型已损坏")
            break
            
    except Exception as e:
        err = str(e).split("\\n")[0][:100]
        print(f"ERROR:step={step} {err}")
        break

# 最终 generate 测试
try:
    test_ids = torch.randint(0, 1000, (1, 32), device=device)
    gen = model.generate(test_ids, max_new_tokens=20, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    torch.cuda.synchronize()
    print("GEN:OK")
except Exception as e:
    err = str(e).split("\\n")[0][:100]
    print(f"GEN:FAIL {err}")
'''

worker_path = "/tmp/_test_2step.py"
with open(worker_path, "w") as f:
    f.write(WORKER)

# ============================================================
# 测试矩阵
# ============================================================
TESTS = [
    # label, use_deepspeed, use_detect_anomaly, use_prepare_kbit, num_steps, seq_len
    ("A: 基线 (无ds, 无anomaly, 有kbit, 3步, seq512)",     False, False, True,  3, 512),
    ("B: 加 detect_anomaly",                              False, True,  True,  3, 512),
    ("C: 不用 prepare_kbit",                              False, False, False, 3, 512),
    ("D: 长序列 seq2048",                                  False, False, True,  3, 2048),
    ("E: 5步看是否逐渐恶化",                               False, False, True,  5, 512),
    ("F: 大学习率 (通过环境变量无法传，这里固定5e-7)",       False, False, True,  3, 512),
]

import os

for label, use_ds, use_da, use_pkb, steps, sl in TESTS:
    cfg = json.dumps({
        "use_deepspeed": use_ds,
        "use_detect_anomaly": use_da,
        "use_prepare_kbit": use_pkb,
        "num_steps": steps,
        "seq_len": sl,
    })
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    env = os.environ.copy()
    
    try:
        result = subprocess.run(
            [sys.executable, worker_path, cfg],
            capture_output=True, text=True, timeout=600, env=env,
        )
        
        output = result.stdout + result.stderr
        
        for line in output.split("\n"):
            if line.startswith(("CHECK:", "STEP:", "DEAD:", "GEN:", "ERROR:")):
                tag = line.split(":")[0]
                data_str = line[len(tag)+1:]
                
                if tag == "CHECK":
                    d = json.loads(data_str)
                    status = "✅" if (d["lora_nan"]==0 and d["lora_inf"]==0 and not d["logits_nan"] and not d["logits_inf"] and d["probs_ok"]) else "❌"
                    print(f"  {status} [{d['label']}] lora_nan={d['lora_nan']} lora_inf={d['lora_inf']} lora_max={d['lora_max']} "
                          f"logits_nan={d['logits_nan']} logits_inf={d['logits_inf']} logits_max={d['logits_max']} probs_ok={d['probs_ok']}")
                elif tag == "STEP":
                    d = json.loads(data_str)
                    flag = "⚠️" if (d["grad_nan"]>0 or d["grad_inf"]>0 or d["loss_nan"]) else "  "
                    print(f"  {flag} Step {d['step']}: loss={d['loss']} grad_nan={d['grad_nan']} grad_inf={d['grad_inf']} "
                          f"grad_max_before={d['grad_max_before_clip']} grad_max_after={d['grad_max_after_clip']}")
                elif tag == "DEAD":
                    print(f"  💀 {data_str}")
                elif tag == "GEN":
                    print(f"  🔄 generate: {data_str}")
                elif tag == "ERROR":
                    print(f"  ❌ {data_str}")
                    
        if result.returncode != 0:
            # 检查是否有 CUDA 错误
            for line in result.stderr.split("\n"):
                if "CUDA error" in line or "assert" in line.lower() or "probability tensor" in line:
                    print(f"  💀 {line.strip()[:100]}")
                    
    except subprocess.TimeoutExpired:
        print(f"  ⏰ 超时")

print(f"\n{'='*70}")
print("诊断完成")
print(f"{'='*70}")