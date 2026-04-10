"""
测试 GRPO 实际差异：极端 advantage + 无梯度裁剪（模拟 DeepSpeed 缺失 gradient_clipping）
用法: CUDA_LAUNCH_BLOCKING=1 LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=1 python diagnose_grpo_loss.py
"""
import subprocess, sys, os, json

WORKER = r'''
import torch, json, sys, gc

MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"
device = "cuda:0"

cfg = json.loads(sys.argv[1])
use_grad_clip = cfg["use_grad_clip"]
clip_value = cfg["clip_value"]
loss_scale = cfg["loss_scale"]   # 模拟 GRPO advantage 放大 loss
num_steps = cfg["num_steps"]
seq_len = cfg["seq_len"]
use_accum = cfg["use_accum"]     # 模拟 gradient accumulation
accum_steps = cfg["accum_steps"]

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    attn_implementation="sdpa", low_cpu_mem_usage=True,
)
if hasattr(model.config, "sliding_window"):
    model.config.sliding_window = None
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

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

def check_health():
    test_ids = torch.randint(0, 1000, (1, 64), device=device)
    with torch.no_grad():
        out = model(input_ids=test_ids)
    logits = out.logits[0, -1, :]
    logits_nan = torch.isnan(logits).any().item()
    logits_inf = torch.isinf(logits).any().item()
    logits_max = logits.abs().max().item() if not (logits_nan or logits_inf) else float('inf')
    
    lora_max = 0.0
    lora_nan = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if torch.isnan(p.data).any(): lora_nan += 1
            m = p.data.abs().max().item()
            if m > lora_max: lora_max = m
    
    probs = torch.softmax(logits.float(), dim=-1)
    probs_ok = not (torch.isnan(probs).any().item() or (probs < 0).any().item())
    
    healthy = lora_nan == 0 and not logits_nan and not logits_inf and probs_ok
    return {
        "healthy": healthy, "lora_nan": lora_nan, "lora_max": round(lora_max, 6),
        "logits_nan": logits_nan, "logits_inf": logits_inf,
        "logits_max": round(logits_max, 2) if logits_max < 1e10 else "inf",
        "probs_ok": probs_ok,
    }

h = check_health()
print("CHECK:init:" + json.dumps(h))

for step in range(1, num_steps + 1):
    optimizer.zero_grad()
    
    total_loss = 0.0
    micro_steps = accum_steps if use_accum else 1
    
    for micro in range(micro_steps):
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
        labels = input_ids.clone()
        
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        
        # 模拟 GRPO: loss 乘以 advantage（可能很大或很小甚至是负的）
        scaled_loss = loss * loss_scale
        
        if use_accum:
            scaled_loss = scaled_loss / accum_steps
        
        scaled_loss.backward()
        torch.cuda.synchronize()
        total_loss += loss.item()
    
    # 梯度统计
    grad_max = 0.0
    grad_nan = 0
    grad_inf = 0
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any(): grad_nan += 1
            if torch.isinf(p.grad).any(): grad_inf += 1
            gm = p.grad.abs().max().item()
            if gm > grad_max: grad_max = gm
    
    # 梯度裁剪
    if use_grad_clip:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=clip_value
        )
    
    grad_max_after = 0.0
    for p in model.parameters():
        if p.grad is not None:
            gm = p.grad.abs().max().item()
            if gm > grad_max_after: grad_max_after = gm
    
    optimizer.step()
    torch.cuda.synchronize()
    
    avg_loss = total_loss / micro_steps
    print(f"STEP:{json.dumps({'step':step,'loss':round(avg_loss,4),'grad_nan':grad_nan,'grad_inf':grad_inf,'grad_max_before':round(grad_max,4),'grad_max_after':round(grad_max_after,6)})}")
    
    h = check_health()
    print(f"CHECK:step{step}:" + json.dumps(h))
    
    if not h["healthy"]:
        print("DEAD")
        break

# 最终 generate
try:
    test_ids = torch.randint(0, 1000, (1, 32), device=device)
    with torch.no_grad():
        gen = model.generate(test_ids, max_new_tokens=20, do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    torch.cuda.synchronize()
    print("GEN:OK")
except Exception as e:
    print(f"GEN:FAIL:{str(e)[:80]}")
'''

worker_path = "/tmp/_test_grpo.py"
with open(worker_path, "w") as f:
    f.write(WORKER)

TESTS = [
    # label, grad_clip, clip_val, loss_scale, steps, seq_len, use_accum, accum_steps
    ("A: 基线 (有裁剪, scale=1)",            True,  0.3,   1.0,  3, 512, False, 1),
    ("B: 无裁剪, scale=1",                   False, 0.3,   1.0,  3, 512, False, 1),
    ("C: 无裁剪, scale=5 (模拟大 advantage)", False, 0.3,   5.0,  3, 512, False, 1),
    ("D: 无裁剪, scale=20",                   False, 0.3,  20.0,  3, 512, False, 1),
    ("E: 无裁剪, scale=50",                   False, 0.3,  50.0,  3, 512, False, 1),
    ("F: 无裁剪, scale=-5 (负 advantage)",    False, 0.3,  -5.0,  3, 512, False, 1),
    ("G: 无裁剪, scale=20, accum=4",          False, 0.3,  20.0,  3, 512, True,  4),
    ("H: 有裁剪, scale=20",                   True,  0.3,  20.0,  3, 512, False, 1),
    ("I: 有裁剪, scale=50",                   True,  0.3,  50.0,  3, 512, False, 1),
    ("J: 无裁剪, scale=100",                  False, 0.3, 100.0,  3, 512, False, 1),
]

for label, gc, cv, ls, ns, sl, ua, acs in TESTS:
    cfg = json.dumps({
        "use_grad_clip": gc, "clip_value": cv, "loss_scale": ls,
        "num_steps": ns, "seq_len": sl, "use_accum": ua, "accum_steps": acs,
    })
    
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, worker_path, cfg],
            capture_output=True, text=True, timeout=600, env=os.environ.copy(),
        )
        
        output = result.stdout
        for line in output.split("\n"):
            if line.startswith("CHECK:"):
                parts = line.split(":", 2)
                stage = parts[1]
                data = json.loads(parts[2])
                status = "✅" if data["healthy"] else "❌"
                print(f"  {status} [{stage:>6}] lora_max={data['lora_max']} logits_max={data['logits_max']} "
                      f"logits_nan={data['logits_nan']} probs_ok={data['probs_ok']}")
            elif line.startswith("STEP:"):
                d = json.loads(line[5:])
                flag = "⚠️" if (d["grad_nan"]>0 or d["grad_inf"]>0) else "  "
                print(f"  {flag} Step {d['step']}: loss={d['loss']:>8} grad_max_before={d['grad_max_before']:>8} "
                      f"grad_max_after={d['grad_max_after']:>10} grad_nan={d['grad_nan']} grad_inf={d['grad_inf']}")
            elif line.startswith("DEAD"):
                print(f"  💀 模型损坏")
            elif line.startswith("GEN:"):
                print(f"  🔄 generate: {line[4:]}")
                
        if result.returncode != 0 and "GEN:" not in result.stdout:
            for line in result.stderr.split("\n"):
                if "CUDA error" in line or "probability tensor" in line:
                    print(f"  💀 {line.strip()[:90]}")
                    
    except subprocess.TimeoutExpired:
        print(f"  ⏰ 超时")

print(f"\n{'='*70}")
print("诊断完成")
print(f"{'='*70}")