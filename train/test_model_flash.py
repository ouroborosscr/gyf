"""
通过实际模型 forward 测试 flash_attn 崩溃点
裸调用全部通过，所以问题在 transformers 的调用方式

用法: CUDA_LAUNCH_BLOCKING=1 CUDA_HOME=/usr/local/cuda-12.8 \
      LD_PRELOAD=/opt/anaconda3/envs/scr_train2/lib/libstdc++.so.6 \
      CUDA_VISIBLE_DEVICES=2 python test_model_flash.py
"""
import torch, gc

device = "cuda:0"
MODEL_PATH = "/date/sunchengrui/models/Qwen3.5-9B"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
)

print("=" * 70)
print("加载模型 (flash_attention_2)...")
print("=" * 70)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, device_map={"": 0},
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
)
if hasattr(model.config, "sliding_window"):
    print(f"  sliding_window 原始值: {model.config.sliding_window}")
    model.config.sliding_window = None
    print(f"  sliding_window 已设为 None")

model.eval()

# ============================================================
print("\n" + "=" * 70)
print("[Test 1] model.forward (no_grad) - 逐步增加序列长度")
print("=" * 70)

for seq_len in [256, 512, 1024, 2048, 3072, 4096, 5120, 6144]:
    gc.collect()
    torch.cuda.empty_cache()
    
    input_ids = torch.randint(100, 5000, (1, seq_len), device=device)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        print(f"  seq={seq_len:>5}: ✅ logits shape={out.logits.shape}")
        del out
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  seq={seq_len:>5}: ❌ {err}")
        # CUDA error 后所有后续操作都会失败，需要重新测试
        break
    
    del input_ids

# ============================================================
print("\n" + "=" * 70)
print("[Test 2] model.forward (with grad) - 模拟 training forward")
print("=" * 70)

for seq_len in [256, 512, 1024, 2048, 3072, 4096]:
    gc.collect()
    torch.cuda.empty_cache()
    
    input_ids = torch.randint(100, 5000, (1, seq_len), device=device)
    try:
        out = model(input_ids=input_ids, use_cache=False)
        torch.cuda.synchronize()
        print(f"  seq={seq_len:>5}: ✅ logits shape={out.logits.shape}")
        del out
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  seq={seq_len:>5}: ❌ {err}")
        break
    
    del input_ids

# ============================================================
print("\n" + "=" * 70)
print("[Test 3] forward+backward → 再 forward (模拟 Step1→Step2)")
print("=" * 70)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 需要重新加载模型因为上面可能已经 CUDA error
gc.collect()
torch.cuda.empty_cache()

# 检查 CUDA 是否还健康
try:
    test = torch.ones(1, device=device)
    del test
    cuda_ok = True
except:
    cuda_ok = False
    print("  ⚠️ CUDA 已经不健康，跳过 Test 3")

if cuda_ok:
    print("  重新加载模型...")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
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
    
    seq_len = 2048  # 用中等长度
    
    # Step 1: forward + backward
    print(f"  Step 1: forward+backward (seq={seq_len})...")
    input_ids = torch.randint(100, 5000, (1, seq_len), device=device)
    try:
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[:, :-1, :]
        target = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1)
        )
        loss.backward()
        torch.cuda.synchronize()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  Step 1: ✅ loss={loss.item():.4f}")
        
        del out, logits, target, loss
        gc.collect()
        torch.cuda.empty_cache()
        
        # Step 2: 只做 forward (no_grad)，模拟 ref model forward
        print(f"  Step 2: ref forward (no_grad, seq={seq_len})...")
        input_ids2 = torch.randint(100, 5000, (1, seq_len), device=device)
        with torch.no_grad():
            out2 = model(input_ids=input_ids2, use_cache=False)
        torch.cuda.synchronize()
        print(f"  Step 2: ✅ logits shape={out2.logits.shape}")
        del out2, input_ids2
        
        # Step 2b: 更长序列的 ref forward
        for ref_len in [3072, 4096, 5120]:
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"  Step 2b: ref forward (no_grad, seq={ref_len})...")
            input_ids3 = torch.randint(100, 5000, (1, ref_len), device=device)
            try:
                with torch.no_grad():
                    out3 = model(input_ids=input_ids3, use_cache=False)
                torch.cuda.synchronize()
                print(f"  Step 2b: ✅ seq={ref_len}")
                del out3
            except Exception as e:
                err = str(e).split('\n')[0][:80]
                print(f"  Step 2b: ❌ seq={ref_len}: {err}")
                break
            del input_ids3
        
    except Exception as e:
        err = str(e).split('\n')[0][:80]
        print(f"  ❌ {err}")

# ============================================================
print("\n" + "=" * 70)
print("[Test 4] 检查 transformers 传给 flash_attn 的实际参数")
print("=" * 70)

# 通过 monkey-patch 拦截 flash_attn 调用
import flash_attn.flash_attn_interface as fai

_original_varlen = fai.flash_attn_varlen_func
_call_count = 0

def _intercepted_varlen(*args, **kwargs):
    global _call_count
    _call_count += 1
    if _call_count <= 3:  # 只打印前几次
        q = args[0] if len(args) > 0 else kwargs.get('q')
        k = args[1] if len(args) > 1 else kwargs.get('k')
        cu_q = args[3] if len(args) > 3 else kwargs.get('cu_seqlens_q')
        cu_k = args[4] if len(args) > 4 else kwargs.get('cu_seqlens_k')
        max_sq = args[5] if len(args) > 5 else kwargs.get('max_seqlen_q')
        max_sk = args[6] if len(args) > 6 else kwargs.get('max_seqlen_k')
        
        print(f"\n  📌 flash_attn_varlen_func 调用 #{_call_count}:")
        print(f"    q: shape={q.shape}, dtype={q.dtype}, device={q.device}")
        print(f"    k: shape={k.shape}, dtype={k.dtype}")
        print(f"    cu_seqlens_q: {cu_q}")
        print(f"    cu_seqlens_k: {cu_k}")
        print(f"    max_seqlen_q: {max_sq}")
        print(f"    max_seqlen_k: {max_sk}")
        
        # 检查参数合法性
        total_q = q.shape[0]
        expected_total = cu_q[-1].item()
        if total_q != expected_total:
            print(f"    ⚠️ q.shape[0]={total_q} != cu_seqlens_q[-1]={expected_total}")
        
        for kw, val in kwargs.items():
            if kw not in ['q', 'k', 'v', 'cu_seqlens_q', 'cu_seqlens_k', 'max_seqlen_q', 'max_seqlen_k']:
                print(f"    {kw}: {val}")
    
    return _original_varlen(*args, **kwargs)

# 安装拦截器
fai.flash_attn_varlen_func = _intercepted_varlen

# 如果 CUDA 还健康，做一次 model forward 看参数
try:
    test = torch.ones(1, device=device)
    del test
    
    print("  通过模型 forward 触发 flash_attn_varlen_func...")
    input_ids = torch.randint(100, 5000, (1, 1024), device=device)
    with torch.no_grad():
        # 需要重新加载 model (之前可能 LoRA 化了)
        out = model(input_ids=input_ids, use_cache=False)
    torch.cuda.synchronize()
    print(f"\n  ✅ 完成，共调用 flash_attn_varlen_func {_call_count} 次")
except Exception as e:
    print(f"  ❌ {str(e).split(chr(10))[0][:80]}")

# 恢复
fai.flash_attn_varlen_func = _original_varlen

print(f"\n{'='*70}")
print("测试完成")
print(f"{'='*70}")