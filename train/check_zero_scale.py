"""
查 ZeROOptimizer.scale_if_loss 在 bf16 模式下的实际行为
用法: LD_PRELOAD=... python check_zero_scale.py
"""
import inspect

print("=" * 70)
print("1. 找所有 scale_if_loss 实现")
print("=" * 70)

# ZeRO Stage 1+2
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

if hasattr(DeepSpeedZeroOptimizer, 'scale_if_loss'):
    src = inspect.getsource(DeepSpeedZeroOptimizer.scale_if_loss)
    print(f"\nDeepSpeedZeroOptimizer.scale_if_loss:")
    print(src)

# ZeRO Stage 3
try:
    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
    if hasattr(DeepSpeedZeroOptimizer_Stage3, 'scale_if_loss'):
        src = inspect.getsource(DeepSpeedZeroOptimizer_Stage3.scale_if_loss)
        print(f"\nDeepSpeedZeroOptimizer_Stage3.scale_if_loss:")
        print(src)
except:
    pass

# BF16 Optimizer
try:
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    if hasattr(BF16_Optimizer, 'scale_if_loss'):
        src = inspect.getsource(BF16_Optimizer.scale_if_loss)
        print(f"\nBF16_Optimizer.scale_if_loss:")
        print(src)
except:
    pass

print("\n" + "=" * 70)
print("2. loss_scaler / external_loss_scale 初始化")
print("=" * 70)

src = inspect.getsource(DeepSpeedZeroOptimizer.__init__)
lines = src.split('\n')
for i, line in enumerate(lines):
    if any(kw in line for kw in ['loss_scale', 'loss_scaler', 'external_loss', 'dynamic_loss', 'cur_scale']):
        context_start = max(0, i-1)
        context_end = min(len(lines), i+2)
        for j in range(context_start, context_end):
            print(f"  {j}: {lines[j]}")
        print()

print("\n" + "=" * 70)
print("3. LossScaler 类")
print("=" * 70)

# 搜索 LossScaler
try:
    from deepspeed.runtime.fp16.loss_scaler import LossScaler, DynamicLossScaler, CreateLossScaler
    
    if hasattr(LossScaler, 'loss_scale'):
        print(f"  LossScaler.loss_scale: {inspect.getsource(LossScaler.__init__)[:500]}")
    
    src = inspect.getsource(DynamicLossScaler.__init__)
    print(f"\n  DynamicLossScaler.__init__:")
    print(src)
except Exception as e:
    print(f"  {e}")

print("\n" + "=" * 70)
print("4. DeepSpeed engine 中 optimizer 的创建 (bf16 路径)")
print("=" * 70)

from deepspeed.runtime.engine import DeepSpeedEngine

# 搜索 _configure_optimizer 或类似方法
for method in ['_configure_optimizer', '_configure_zero_optimizer']:
    if hasattr(DeepSpeedEngine, method):
        src = inspect.getsource(getattr(DeepSpeedEngine, method))
        lines = src.split('\n')
        print(f"\n  {method} 中与 loss_scale/bf16 相关的行:")
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['loss_scale', 'bf16', 'bfloat', 'external_loss', 'dynamic_loss']):
                if line.strip() and not line.strip().startswith('#'):
                    print(f"    {i}: {line.strip()}")

print("\n" + "=" * 70)
print("5. 关键问题：bf16=True 时 loss_scale 是多少？")
print("=" * 70)

# 搜索默认值
try:
    from deepspeed.runtime.constants import (
        INITIAL_LOSS_SCALE, INITIAL_LOSS_SCALE_DEFAULT,
    )
    print(f"  INITIAL_LOSS_SCALE_DEFAULT = {INITIAL_LOSS_SCALE_DEFAULT}")
except ImportError as e:
    print(f"  无法导入 INITIAL_LOSS_SCALE_DEFAULT: {e}")
    # 手动搜索
    import deepspeed.runtime.constants as c
    for name in sorted(dir(c)):
        val = getattr(c, name)
        if isinstance(val, (int, float)) and 'SCALE' in name.upper():
            print(f"  {name} = {val}")

# 搜索 bf16 配置中是否禁用了 loss scaling
print("\n  搜索 bf16 配置中 loss_scale 的处理:")
for method in ['_configure_optimizer', '_configure_zero_optimizer', '_do_optimizer_sanity_check']:
    if hasattr(DeepSpeedEngine, method):
        src = inspect.getsource(getattr(DeepSpeedEngine, method))
        if 'bf16' in src and 'loss_scale' in src:
            lines = src.split('\n')
            for i, line in enumerate(lines):
                if ('bf16' in line or 'loss_scale' in line) and line.strip():
                    print(f"    [{method}] {i}: {line.strip()}")

print("\n" + "=" * 70)
print("6. 直接检查: bf16 + ZeRO-2 时，optimizer 类型是什么")
print("=" * 70)

# 读 _configure_zero_optimizer 完整代码
if hasattr(DeepSpeedEngine, '_configure_zero_optimizer'):
    src = inspect.getsource(DeepSpeedEngine._configure_zero_optimizer)
    # 找 bf16/bfloat 相关分支
    lines = src.split('\n')
    in_bf16_block = False
    for i, line in enumerate(lines):
        if 'bf16' in line.lower() or 'bfloat' in line.lower():
            in_bf16_block = True
        if in_bf16_block:
            print(f"  {i}: {line}")
            if line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                # 检测到新的 if/elif/else 块结束
                if i > 0 and (line.strip().startswith('elif') or line.strip().startswith('else') or line.strip().startswith('return')):
                    if 'bf16' not in line.lower():
                        break
        if i > 300:  # 安全限制
            break

print("\n" + "=" * 70)
print("完成")
print("=" * 70)