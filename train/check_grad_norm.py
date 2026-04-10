"""
检查 max_grad_norm 在 TRL GRPOTrainer + DeepSpeed 下是否真正生效
用法: python check_grad_norm.py
"""
import inspect

print("=" * 70)
print("1. GRPOConfig 中 max_grad_norm 的值")
print("=" * 70)

from trl import GRPOConfig
config = GRPOConfig(output_dir="/tmp/test", max_grad_norm=0.3)
print(f"  config.max_grad_norm = {config.max_grad_norm}")

# GRPOConfig 继承链
print(f"\n  GRPOConfig MRO:")
for cls in GRPOConfig.__mro__:
    print(f"    {cls.__name__}")

# ============================================================
print("\n" + "=" * 70)
print("2. max_grad_norm 在 Trainer 中哪里被使用")
print("=" * 70)

from transformers import Trainer

# 搜索 training_step 和相关方法中对 max_grad_norm 的引用
for method_name in ['training_step', '_inner_training_loop', '_run_epoch']:
    if hasattr(Trainer, method_name):
        src = inspect.getsource(getattr(Trainer, method_name))
        if 'max_grad_norm' in src or 'grad_norm' in src or 'clip_grad' in src:
            print(f"\n  [{method_name}] 包含梯度裁剪相关代码:")
            lines = src.split('\n')
            for i, line in enumerate(lines):
                if any(kw in line for kw in ['max_grad_norm', 'clip_grad', 'grad_norm', 'gradient_clipping']):
                    start = max(0, i-1)
                    end = min(len(lines), i+2)
                    for j in range(start, end):
                        print(f"    {j}: {lines[j]}")
                    print()

# ============================================================
print("\n" + "=" * 70)
print("3. GRPOTrainer 是否覆盖了 training_step")
print("=" * 70)

from trl.trainer.grpo_trainer import GRPOTrainer

if 'training_step' in GRPOTrainer.__dict__:
    src = inspect.getsource(GRPOTrainer.training_step)
    print("  GRPOTrainer 覆盖了 training_step:")
    if 'max_grad_norm' in src or 'clip_grad' in src:
        print("  → 包含梯度裁剪代码")
    else:
        print("  → 不包含梯度裁剪代码（依赖父类 Trainer）")
    # 打印关键部分
    lines = src.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#'):
            print(f"    {i}: {stripped}")
else:
    print("  GRPOTrainer 没有覆盖 training_step（使用 Trainer 的）")

# ============================================================
print("\n" + "=" * 70)
print("4. DeepSpeed 下梯度裁剪的走法")
print("=" * 70)

# Trainer 在 DeepSpeed 模式下如何处理 gradient clipping
src = inspect.getsource(Trainer.training_step)
print("  Trainer.training_step 中与 DeepSpeed/梯度相关的代码:")
lines = src.split('\n')
for i, line in enumerate(lines):
    if any(kw in line.lower() for kw in ['deepspeed', 'clip', 'grad_norm', 'accelerator.clip', 'max_grad_norm']):
        start = max(0, i-2)
        end = min(len(lines), i+3)
        for j in range(start, end):
            print(f"  {j}: {lines[j]}")
        print()

# ============================================================
print("\n" + "=" * 70)
print("5. accelerator.clip_grad_norm_ 实现")
print("=" * 70)

from accelerate import Accelerator
if hasattr(Accelerator, 'clip_grad_norm_'):
    src = inspect.getsource(Accelerator.clip_grad_norm_)
    print("  Accelerator.clip_grad_norm_:")
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if any(kw in line.lower() for kw in ['deepspeed', 'clip', 'grad_norm', 'max_norm', 'return']):
            print(f"    {i}: {line}")

# ============================================================
print("\n" + "=" * 70)
print("6. DeepSpeed engine 的 gradient_clipping")
print("=" * 70)

try:
    import deepspeed
    print(f"  DeepSpeed 版本: {deepspeed.__version__}")
    
    # 检查 DeepSpeed engine 的 clip_grad 方法
    from deepspeed.runtime.engine import DeepSpeedEngine
    if hasattr(DeepSpeedEngine, '_do_gradient_clipping'):
        src = inspect.getsource(DeepSpeedEngine._do_gradient_clipping)
        print("\n  DeepSpeedEngine._do_gradient_clipping:")
        lines = src.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped:
                print(f"    {i}: {stripped}")
    
    # 检查 config 中 gradient_clipping 的读取
    if hasattr(DeepSpeedEngine, '__init__'):
        src = inspect.getsource(DeepSpeedEngine.__init__)
        if 'gradient_clipping' in src:
            print("\n  DeepSpeedEngine.__init__ 读取 gradient_clipping:")
            lines = src.split('\n')
            for i, line in enumerate(lines):
                if 'gradient_clipping' in line or 'clip' in line.lower():
                    print(f"    {i}: {line.strip()}")
except ImportError:
    print("  DeepSpeed 未安装")

# ============================================================
print("\n" + "=" * 70)
print("7. 你的 ds_config.json 中的 gradient_clipping")
print("=" * 70)

import json, os
ds_path = "ds_config.json"
if os.path.exists(ds_path):
    with open(ds_path) as f:
        ds = json.load(f)
    gc = ds.get("gradient_clipping", "❌ 不存在!")
    print(f"  ds_config.json gradient_clipping = {gc}")
    
    # 检查 GRPOConfig 的 max_grad_norm 是否会覆盖 ds_config
    print(f"\n  关键问题: 当 DeepSpeed 启用时，谁做梯度裁剪？")
    print(f"  → GRPOConfig.max_grad_norm = 0.3")
    print(f"  → ds_config.gradient_clipping = {gc}")
    print(f"  → 如果两者都设了，Trainer 让 DeepSpeed 做裁剪")
    print(f"  → 如果 ds_config 没设，Trainer 可能跳过裁剪（取决于 accelerate 实现）")
else:
    print(f"  ds_config.json 不存在")

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)