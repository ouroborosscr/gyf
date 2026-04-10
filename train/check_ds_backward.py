"""
提取 DeepSpeed engine.backward 的完整实现（绕过装饰器）
用法: python check_ds_backward.py
"""
import inspect
import textwrap

print("=" * 70)
print("1. DeepSpeed backward 完整源码（绕过装饰器）")
print("=" * 70)

from deepspeed.runtime.engine import DeepSpeedEngine

# 尝试多种方式获取真实实现
fn = DeepSpeedEngine.backward

# 解包装饰器
while hasattr(fn, '__wrapped__'):
    fn = fn.__wrapped__

# 如果是 closure，获取内部函数
if hasattr(fn, '__closure__') and fn.__closure__:
    for cell in fn.__closure__:
        try:
            inner = cell.cell_contents
            if callable(inner) and hasattr(inner, '__qualname__'):
                if 'backward' in inner.__qualname__.lower():
                    fn = inner
                    break
        except ValueError:
            pass

try:
    src = inspect.getsource(fn)
    print(src)
except:
    print("  无法获取源码")

# ============================================================
print("\n" + "=" * 70)
print("2. 直接读 DeepSpeed 源码文件")
print("=" * 70)

import deepspeed
ds_path = inspect.getfile(DeepSpeedEngine)
print(f"  文件路径: {ds_path}")

# 读文件找 def backward
with open(ds_path, 'r') as f:
    content = f.read()

lines = content.split('\n')
in_backward = False
indent_level = None
backward_lines = []

for i, line in enumerate(lines):
    if 'def backward(' in line and 'self' in line:
        in_backward = True
        indent_level = len(line) - len(line.lstrip())
        backward_lines.append(f"{i+1:>5}: {line}")
        continue
    
    if in_backward:
        if line.strip() and not line.strip().startswith('#'):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and line.strip().startswith('def '):
                break
        backward_lines.append(f"{i+1:>5}: {line}")

print(f"\n  backward 方法 ({len(backward_lines)} 行):")
for line in backward_lines[:80]:  # 最多打印80行
    print(f"  {line}")

# ============================================================
print("\n" + "=" * 70)
print("3. 搜索 backward 中的 scale/multiply/accumulate 操作")
print("=" * 70)

for line in backward_lines:
    stripped = line.strip()
    # 去掉行号
    content_part = stripped.split(':', 1)[1] if ':' in stripped else stripped
    content_part = content_part.strip()
    if any(kw in content_part.lower() for kw in [
        'scale', 'multiply', '*=', 'accum', 'gas', 'loss', 'grad',
        'backward', 'step', 'micro_batch', 'reduce'
    ]):
        if content_part and not content_part.startswith('#') and not content_part.startswith('"""'):
            print(f"  {line}")

# ============================================================
print("\n" + "=" * 70)
print("4. 检查 gradient_accumulation_steps 相关的 scale 方法")
print("=" * 70)

# 打印 _scale_loss_by_gas 完整代码
src = inspect.getsource(DeepSpeedEngine._scale_loss_by_gas)
print("_scale_loss_by_gas:")
print(src)

# 检查有没有反向操作（乘以 gas）
for name in dir(DeepSpeedEngine):
    if callable(getattr(DeepSpeedEngine, name, None)):
        try:
            msrc = inspect.getsource(getattr(DeepSpeedEngine, name))
            if 'gradient_accumulation_steps' in msrc and ('*' in msrc or 'multiply' in msrc.lower()):
                if name not in ['_scale_loss_by_gas', 'backward', '__init__']:
                    print(f"\n  方法 {name} 中使用了 gradient_accumulation_steps 和乘法:")
                    mlines = msrc.split('\n')
                    for ml in mlines:
                        if 'gradient_accumulation_steps' in ml or ('* ' in ml and 'gas' in ml.lower()):
                            print(f"    {ml.strip()}")
        except:
            pass

# ============================================================
print("\n" + "=" * 70)
print("5. 检查 DeepSpeed allreduce 是否做了 scale")
print("=" * 70)

for name in ['allreduce_gradients', '_reduce_non_expert_gradients', 'buffered_allreduce_fallback']:
    if hasattr(DeepSpeedEngine, name):
        try:
            src = inspect.getsource(getattr(DeepSpeedEngine, name))
            if 'scale' in src.lower() or 'divide' in src.lower() or '/ ' in src:
                print(f"\n  {name} 中有 scale 操作:")
                for line in src.split('\n'):
                    if 'scale' in line.lower() or '/' in line or 'divide' in line.lower():
                        print(f"    {line.strip()}")
        except:
            pass

print("\n" + "=" * 70)
print("检查完成")
print("=" * 70)