"""
查 DeepSpeed backward 是否对 loss 做了缩放
用法: python check_ds_loss_scale.py
"""
import inspect

# ============================================================
print("=" * 70)
print("1. DeepSpeed engine.backward 做了什么")
print("=" * 70)

import deepspeed
from deepspeed.runtime.engine import DeepSpeedEngine

src = inspect.getsource(DeepSpeedEngine.backward)
# 只打印关键行
lines = src.split('\n')
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(kw in stripped.lower() for kw in ['scale', 'loss', 'backward', 'grad', 'accum', 'micro', 'gas']):
        if not stripped.startswith('#') and not stripped.startswith('"""') and stripped:
            print(f"  {i:>4}: {stripped[:120]}")

# ============================================================
print("\n" + "=" * 70)
print("2. DeepSpeed _scale_loss_by_gas 方法")
print("=" * 70)

if hasattr(DeepSpeedEngine, '_scale_loss_by_gas'):
    src = inspect.getsource(DeepSpeedEngine._scale_loss_by_gas)
    print(src)
elif hasattr(DeepSpeedEngine, 'scale_loss'):
    src = inspect.getsource(DeepSpeedEngine.scale_loss)
    print(src)
else:
    # 搜索所有方法中包含 scale 和 loss 的
    for name in dir(DeepSpeedEngine):
        if 'scale' in name.lower() and 'loss' in name.lower():
            print(f"  找到方法: {name}")
            src = inspect.getsource(getattr(DeepSpeedEngine, name))
            print(src[:500])

# ============================================================
print("\n" + "=" * 70)
print("3. Trainer.training_step 中 DeepSpeed 的 loss 处理")
print("=" * 70)

from transformers import Trainer

src = inspect.getsource(Trainer.training_step)
lines = src.split('\n')
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(kw in stripped.lower() for kw in ['deepspeed', 'scale', 'backward', 'gas', 'accum']):
        if stripped and not stripped.startswith('#'):
            context_start = max(0, i-1)
            context_end = min(len(lines), i+2)
            for j in range(context_start, context_end):
                print(f"  {j:>4}: {lines[j]}")
            print()

# ============================================================
print("\n" + "=" * 70)
print("4. accelerate backward 中 DeepSpeed 的处理")
print("=" * 70)

from accelerate import Accelerator
src = inspect.getsource(Accelerator.backward)
lines = src.split('\n')
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(kw in stripped.lower() for kw in ['deepspeed', 'scale', 'gas', 'backward']):
        if stripped and not stripped.startswith('#'):
            context_start = max(0, i-2)
            context_end = min(len(lines), i+3)
            for j in range(context_start, context_end):
                print(f"  {j:>4}: {lines[j]}")
            print()

# ============================================================
print("\n" + "=" * 70)
print("5. DeepSpeedEngineWrapper backward")
print("=" * 70)

try:
    from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
    if hasattr(DeepSpeedEngineWrapper, 'backward'):
        src = inspect.getsource(DeepSpeedEngineWrapper.backward)
        print(src)
except:
    print("  无法导入 DeepSpeedEngineWrapper")

# ============================================================
print("\n" + "=" * 70)
print("6. 完整的 DeepSpeed backward 方法")
print("=" * 70)

src = inspect.getsource(DeepSpeedEngine.backward)
print(src[:3000])

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)