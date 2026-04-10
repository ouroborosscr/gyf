"""
直接读 DeepSpeed 源码文件找 backward 方法
"""
import deepspeed, inspect

from deepspeed.runtime.engine import DeepSpeedEngine
ds_path = inspect.getfile(DeepSpeedEngine)
print(f"文件: {ds_path}\n")

with open(ds_path, 'r') as f:
    lines = f.readlines()

# 找 def backward
for i, line in enumerate(lines):
    if 'def backward(' in line:
        # 打印前后 80 行
        start = i
        for j in range(start, min(start + 80, len(lines))):
            print(f"{j+1:>5}: {lines[j]}", end='')
            # 检测下一个 def 开始则停止
            if j > start and lines[j].strip().startswith('def ') and 'backward' not in lines[j]:
                break
        print("\n--- END ---\n")