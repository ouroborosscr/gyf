"""
逐步加变量测试 - 每个配置独立子进程，互不干扰
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_model_layers_v2.py
"""
import subprocess, sys, os, json

TESTS = [
    # label, 4bit, prepare_kbit, lora, gc, flash
    ("A: bf16 + flash (基线)",              False, False, False, False, True),
    ("B: bf16 + flash + gc",               False, False, False, True,  True),
    ("C: bf16 + SDPA (对照)",              False, False, False, False, False),
    ("D: bf16 + SDPA + gc",               False, False, False, True,  False),
    ("E: 4bit + flash",                    True,  False, False, False, True),
    ("F: 4bit + SDPA",                     True,  False, False, False, False),
    ("G: 4bit + prepare_kbit + flash",     True,  True,  False, False, True),
    ("H: 4bit + prepare_kbit + SDPA",      True,  True,  False, False, False),
    ("I: 4bit + prepare_kbit + gc + flash", True,  True,  False, True,  True),
    ("J: 4bit + prepare_kbit + gc + SDPA",  True,  True,  False, True,  False),
    ("K: 完整配置 + flash (4bit+kbit+gc+lora)", True, True, True, True, True),
    ("L: 完整配置 + SDPA (4bit+kbit+gc+lora)",  True, True, True, True, False),
]

WORKER_SCRIPT = '''
import torch, json, sys, gc

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

cfg = json.loads(sys.argv[1])
use_4bit = cfg["use_4bit"]
use_prepare_kbit = cfg["use_prepare_kbit"]
use_lora = cfg["use_lora"]
use_gc = cfg["use_gc"]
use_flash = cfg["use_flash"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)
peft_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)

kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": {"": 0},
    "attn_implementation": "flash_attention_2" if use_flash else "sdpa",
    "low_cpu_mem_usage": True,
}
if use_4bit:
    kwargs["quantization_config"] = bnb_config

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

input_ids = torch.randint(0, 1000, (1, 512), device=device)
labels = input_ids.clone()

out = model(input_ids=input_ids, labels=labels)
loss = out.loss
loss_val = loss.item()
loss.backward()
torch.cuda.synchronize()

grad_nans = 0
grad_total = 0
grad_max = 0.0
for n, p in model.named_parameters():
    if p.grad is not None:
        grad_total += 1
        if torch.isnan(p.grad).any():
            grad_nans += 1
        gm = p.grad.abs().max().item()
        if gm > grad_max:
            grad_max = gm

result = {
    "status": "pass" if grad_nans == 0 else "grad_nan",
    "loss": round(loss_val, 4),
    "grad_nan": grad_nans,
    "grad_total": grad_total,
    "grad_max": round(grad_max, 2),
}
print("RESULT:" + json.dumps(result))
'''

# 写入临时 worker 脚本
worker_path = "/tmp/_test_worker.py"
with open(worker_path, "w") as f:
    f.write(WORKER_SCRIPT)

print("=" * 70)
print(f"{'测试':<45} {'结果':<8} {'详情'}")
print("=" * 70)

for label, b4, pkb, lora, gc_flag, flash in TESTS:
    cfg = json.dumps({
        "use_4bit": b4, "use_prepare_kbit": pkb,
        "use_lora": lora, "use_gc": gc_flag, "use_flash": flash,
    })
    
    env = os.environ.copy()
    
    try:
        result = subprocess.run(
            [sys.executable, worker_path, cfg],
            capture_output=True, text=True, timeout=300, env=env,
        )
        
        # 从 stdout 提取结果
        output = result.stdout + result.stderr
        found = False
        for line in output.split("\n"):
            if line.startswith("RESULT:"):
                data = json.loads(line[7:])
                if data["status"] == "pass":
                    print(f"{label:<45} ✅ pass  loss={data['loss']}, grad_max={data['grad_max']}, params={data['grad_total']}")
                else:
                    print(f"{label:<45} ❌ NaN   loss={data['loss']}, nan_params={data['grad_nan']}/{data['grad_total']}")
                found = True
                break
        
        if not found:
            # 从 stderr 提取关键错误
            err = result.stderr
            if "illegal memory access" in err:
                print(f"{label:<45} 💀 CUDA  illegal memory access")
            elif "illegal instruction" in err:
                print(f"{label:<45} 💀 CUDA  illegal instruction")
            elif "nan values" in err.lower():
                print(f"{label:<45} ❌ NaN   (detect_anomaly caught)")
            elif "RuntimeError" in err:
                for line in err.split("\n"):
                    if "RuntimeError" in line:
                        print(f"{label:<45} ❌ err   {line.strip()[:60]}")
                        break
            elif "Error" in err:
                for line in err.split("\n"):
                    if "Error" in line and "Warning" not in line:
                        print(f"{label:<45} ❌ err   {line.strip()[:60]}")
                        break
            else:
                print(f"{label:<45} ❌ ???   exitcode={result.returncode}")
                # 打印最后几行帮助排查
                last_lines = [l for l in err.strip().split("\n") if l.strip()][-3:]
                for l in last_lines:
                    print(f"  {l[:80]}")
                    
    except subprocess.TimeoutExpired:
        print(f"{label:<45} ⏰ 超时  (>300s)")

print("=" * 70)
print("诊断完成")