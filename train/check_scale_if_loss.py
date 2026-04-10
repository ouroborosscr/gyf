"""
查 ZeROOptimizer.scale_if_loss 到底把 loss 乘了多少
"""
import inspect

# ZeRO Stage 2 用的是哪个 optimizer class
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
print("=" * 70)
print("1. DeepSpeedZeroOptimizer.scale_if_loss")
print("=" * 70)

if hasattr(DeepSpeedZeroOptimizer, 'scale_if_loss'):
    src = inspect.getsource(DeepSpeedZeroOptimizer.scale_if_loss)
    print(src)

# 看 loss_scale 相关的初始化
print("\n" + "=" * 70)
print("2. loss_scale 在 __init__ 中的初始化")
print("=" * 70)

src = inspect.getsource(DeepSpeedZeroOptimizer.__init__)
lines = src.split('\n')
for i, line in enumerate(lines):
    if any(kw in line.lower() for kw in ['loss_scale', 'loss_scaler', 'dynamic_loss']):
        print(f"  {i}: {line.strip()}")

# 看 loss_scaler 的值
print("\n" + "=" * 70)
print("3. bf16 模式下 loss_scale 的值")
print("=" * 70)

# 检查 bf16 配置下是否有 loss scaling
from deepspeed.runtime.engine import DeepSpeedEngine
src = inspect.getsource(DeepSpeedEngine.__init__)
lines = src.split('\n')
for i, line in enumerate(lines):
    if any(kw in line.lower() for kw in ['loss_scale', 'loss_scaler', 'bf16', 'bfloat']):
        if line.strip() and not line.strip().startswith('#'):
            print(f"  {i}: {line.strip()}")

# 检查 bf16 optimizer
print("\n" + "=" * 70)
print("4. BF16_Optimizer.scale_if_loss")
print("=" * 70)

try:
    from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
    if hasattr(BF16_Optimizer, 'scale_if_loss'):
        src = inspect.getsource(BF16_Optimizer.scale_if_loss)
        print(src)
    
    # loss_scale 初始化
    src = inspect.getsource(BF16_Optimizer.__init__)
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'loss_scale' in line.lower():
            print(f"  init {i}: {line.strip()}")
except ImportError:
    print("  BF16_Optimizer 不存在")

# 检查 ZeROOptimizer 基类
print("\n" + "=" * 70)
print("5. 检查所有可能的 optimizer 类的 scale_if_loss")
print("=" * 70)

import deepspeed.runtime
import pkgutil, importlib

for importer, modname, ispkg in pkgutil.walk_packages(deepspeed.runtime.__path__, prefix='deepspeed.runtime.'):
    try:
        mod = importlib.import_module(modname)
        for name in dir(mod):
            obj = getattr(mod, name)
            if isinstance(obj, type) and hasattr(obj, 'scale_if_loss'):
                src = inspect.getsource(obj.scale_if_loss)
                if 'loss_scale' in src or 'cur_scale' in src:
                    print(f"\n  {name}.scale_if_loss:")
                    print(src)
    except:
        pass

print("\n" + "=" * 70)
print("6. 直接模拟：bf16 + ZeRO-2 时 loss_scale 是多少")
print("=" * 70)

# 检查默认的 dynamic loss scale 初始值
try:
    from deepspeed.runtime.utils import DynamicLossScaler
    src = inspect.getsource(DynamicLossScaler.__init__)
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'init_scale' in line or 'cur_scale' in line or 'loss_scale' in line:
            print(f"  {i}: {line.strip()}")
except:
    pass

try:
    from deepspeed.runtime.constants import INITIAL_LOSS_SCALE, INITIAL_LOSS_SCALE_DEFAULT
    print(f"\n  INITIAL_LOSS_SCALE_DEFAULT = {INITIAL_LOSS_SCALE_DEFAULT}")
except:
    try:
        from deepspeed.runtime.constants import *
        # 搜索所有包含 LOSS_SCALE 的常量
        import deepspeed.runtime.constants as c
        for name in dir(c):
            if 'LOSS_SCALE' in name or 'SCALE' in name:
                print(f"  {name} = {getattr(c, name)}")
    except:
        pass

print("\n" + "=" * 70)
print("完成")
print("=" * 70)