"""
Reproduce: Qwen3.5 3D position_ids causes silent SDPA gradient explosion (10^27)
and flash_attention_2 CUDA illegal memory access.

Root cause: _is_packed_sequence() in modeling_flash_attention_utils.py
misinterprets Qwen3.5's 3D position_ids [3, batch, seq_len] as packed sequences.

No private data needed — uses random input_ids.
Toggles fix on/off via monkey-patch to demonstrate before/after in one run.

Requirements:
    pip install torch transformers peft bitsandbytes accelerate

Usage:
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python reproduce_position_ids_bug.py
"""

import torch
import gc
import sys
import transformers
import transformers.modeling_flash_attention_utils as flash_utils
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ==================== Config ====================
MODEL_ID = "/date/sunchengrui/models/Qwen3.5-9B"   # change to local path if needed
SEQ_LEN = 512
DEVICE = "cuda:0"


# ==================== Monkey-patch ====================
def _buggy_is_packed_sequence(position_ids, batch_size):
    """Original code (vulnerable): no check for >2D position_ids."""
    if position_ids is None:
        return False
    increasing = (
        torch.arange(position_ids.shape[1], device=position_ids.device)
        + position_ids.min()
    )
    return batch_size == 1 and (increasing - position_ids).abs().sum().bool()


def _fixed_is_packed_sequence(position_ids, batch_size):
    """Patched code: reject >2D position_ids (PR #44911)."""
    if position_ids is None:
        return False
    if position_ids.dim() > 2:
        return False
    increasing = (
        torch.arange(position_ids.shape[1], device=position_ids.device)
        + position_ids.min()
    )
    return batch_size == 1 and (increasing - position_ids).abs().sum().bool()


# ==================== Helpers ====================
def load_model(attn_impl="sdpa"):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if hasattr(model.config, "sliding_window"):
        model.config.sliding_window = None

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.train()
    return model


def measure_gradient(model):
    """Forward + backward on random data, return (status, grad_max)."""
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-7
    )
    opt.zero_grad()

    torch.manual_seed(42)
    input_ids = torch.randint(100, 5000, (1, SEQ_LEN), device=DEVICE)

    # Forward
    try:
        out = model(input_ids=input_ids, use_cache=False)
    except Exception as e:
        return "CRASH", str(e).split("\n")[0][:100]

    # Simple causal-LM loss
    logits = out.logits[:, :-1, :]
    target = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    per_token = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    loss = (-1.5 * per_token).mean()

    # Backward
    try:
        loss.backward()
        torch.cuda.synchronize()
    except Exception as e:
        return "CRASH_BWD", str(e).split("\n")[0][:100]

    grad_max = max(
        (p.grad.abs().max().item() for _, p in model.named_parameters()
         if p.grad is not None and not torch.isnan(p.grad).any()),
        default=0.0,
    )

    opt.zero_grad()
    del out, logits, log_probs, per_token, loss, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    return "OK", grad_max


def cleanup_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def cuda_healthy():
    try:
        _ = torch.ones(1, device=DEVICE)
        return True
    except Exception:
        return False


# ==================== Main ====================
print("=" * 60)
print("Qwen3.5 position_ids bug — reproduction script")
print("=" * 60)
print(f"Python:       {sys.version.split()[0]}")
print(f"PyTorch:      {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"GPU:          {torch.cuda.get_device_name(0)}")

has_flash = False
try:
    import flash_attn
    print(f"flash_attn:   {flash_attn.__version__}")
    has_flash = True
except ImportError:
    print("flash_attn:   not installed (flash tests skipped)")

# ---------- Show Qwen3.5 hybrid architecture ----------
print("\n--- Qwen3.5 layer types ---")
probe = load_model("sdpa")
for i, layer in enumerate(probe.base_model.model.model.layers):
    if i > 7:
        print("  ...")
        break
    kind = "Attention" if hasattr(layer, "self_attn") else "GatedDeltaNet"
    print(f"  Layer {i}: {kind}")
cleanup_model(probe)

# ---------- Define tests ----------
tests = [
    ("A", "sdpa", False, "SDPA  + buggy  → gradient explosion"),
    ("B", "sdpa", True,  "SDPA  + fixed  → normal gradient"),
]
if has_flash:
    # Put crash test last so it doesn't kill CUDA for later tests
    tests += [
        ("C", "flash_attention_2", True,  "flash  + fixed  → normal gradient"),
        ("D", "flash_attention_2", False, "flash  + buggy  → CUDA crash"),
    ]

results = []

for tid, attn, fixed, desc in tests:
    print(f"\n{'=' * 60}")
    print(f"Test {tid}: {desc}")
    print(f"{'=' * 60}")

    # Toggle
    flash_utils._is_packed_sequence = (
        _fixed_is_packed_sequence if fixed else _buggy_is_packed_sequence
    )
    tag = "FIXED" if fixed else "BUGGY"

    # Quick sanity: does _is_packed_sequence misfire on 3D input?
    fake = torch.zeros(3, 1, 8, dtype=torch.long, device=DEVICE)
    verdict = flash_utils._is_packed_sequence(fake, batch_size=1)
    print(f"  _is_packed_sequence([3,1,8]) = {verdict}  ({tag})")
    del fake

    # Load
    print(f"  Loading model ({attn}) ...")
    try:
        model = load_model(attn)
    except Exception as e:
        print(f"  LOAD FAIL: {e!s:.60}")
        results.append((tid, attn, tag, "LOAD_FAIL", "N/A"))
        continue

    # Test
    print(f"  Forward + backward (seq={SEQ_LEN}) ...")
    status, value = measure_gradient(model)
    cleanup_model(model)

    if status == "OK":
        label = "EXPLODED" if value > 1e10 else "normal"
        print(f"  Result: grad_max = {value:.2e}  → {label}")
        results.append((tid, attn, tag, label, f"{value:.2e}"))
    else:
        print(f"  Result: {status} — {value}")
        results.append((tid, attn, tag, status, "N/A"))

    if not cuda_healthy():
        print("  ⚠ CUDA context dead — skipping remaining tests")
        break

# ---------- Summary ----------
print(f"\n{'=' * 60}")
print("Summary")
print(f"{'=' * 60}")
header = f"{'Test':<6}{'Attention':<20}{'Patch':<8}{'Result':<14}{'grad_max'}"
print(header)
print("-" * len(header))
for tid, attn, tag, label, gmax in results:
    print(f"{tid:<6}{attn:<20}{tag:<8}{label:<14}{gmax}")

print(f"""
Root cause
----------
_is_packed_sequence() treats Qwen3.5's 3D position_ids [3, B, S] as a
packed sequence.  It reshapes to [3*B*S], finds 3 zero-positions, and
builds cu_seqlens claiming 3x the actual tokens.

  • SDPA path  : silently produces 10^27 gradients (model destroyed in 2 steps)
  • flash_attn : reads past tensor boundary → CUDA illegal memory access

Fix: add `if position_ids.dim() > 2: return False`
  Issue: https://github.com/huggingface/transformers/issues/44910
  PR:    https://github.com/huggingface/transformers/pull/44911
""")