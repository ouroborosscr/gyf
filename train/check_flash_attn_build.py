"""
检查 flash_attn 的编译信息是否和当前环境匹配
用法: python check_flash_attn_build.py
"""
import subprocess, sys, os

print("=" * 60)
print("1. 当前环境")
print("=" * 60)

import torch
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA (torch):   {torch.version.cuda}")
print(f"CUDA (nvcc):    ", end="")
try:
    nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
    for line in nvcc.stdout.strip().split("\n"):
        if "release" in line.lower():
            print(line.strip())
            break
except:
    print("nvcc 不可用")

print(f"Python:         {sys.version.split()[0]}")

import flash_attn
print(f"flash_attn:     {flash_attn.__version__}")

print("\n" + "=" * 60)
print("2. flash_attn 安装来源")
print("=" * 60)

# pip show
result = subprocess.run([sys.executable, "-m", "pip", "show", "flash-attn"], capture_output=True, text=True)
print(result.stdout)

# 检查是从 wheel 安装还是源码编译
result2 = subprocess.run([sys.executable, "-m", "pip", "show", "-f", "flash-attn"], capture_output=True, text=True)
lines = result2.stdout.strip().split("\n")
# 找 Installer
for line in lines:
    if "Installer:" in line:
        print(f"安装方式: {line.strip()}")

print("\n" + "=" * 60)
print("3. flash_attn .so 文件检查")
print("=" * 60)

import flash_attn
fa_dir = os.path.dirname(flash_attn.__file__)
print(f"flash_attn 目录: {fa_dir}")

# 查找所有 .so 文件
so_files = []
for root, dirs, files in os.walk(fa_dir):
    for f in files:
        if f.endswith(".so"):
            so_files.append(os.path.join(root, f))

for so in so_files:
    print(f"\n  {os.path.basename(so)}")
    # 检查链接的 CUDA 库
    try:
        ldd = subprocess.run(["ldd", so], capture_output=True, text=True)
        for line in ldd.stdout.split("\n"):
            if "cuda" in line.lower() or "cublas" in line.lower() or "not found" in line.lower():
                print(f"    {line.strip()}")
    except:
        pass

    # 检查文件名中的版本信息（wheel 通常编码了 CUDA 版本）
    basename = os.path.basename(so)
    if "cu" in basename.lower():
        print(f"    文件名含 CUDA 版本标识: {basename}")

print("\n" + "=" * 60)
print("4. flash_attn_gpu 模块检查")
print("=" * 60)

try:
    # 这是 flash_attn 的 C++ 扩展模块
    from flash_attn import flash_attn_gpu
    print(f"flash_attn_gpu 位置: {flash_attn_gpu.__file__}")
except ImportError:
    # 新版 flash_attn 可能用 torch 的 custom op
    print("flash_attn_gpu 不是直接导入的（可能用了 torch custom op）")
    
    # 查找实际的 .so
    try:
        import flash_attn._C as _C
        print(f"flash_attn._C 位置: {_C.__file__}")
    except:
        pass

# 检查 torch.ops 里注册的 flash_attn op
try:
    if hasattr(torch.ops, 'flash_attn'):
        print(f"torch.ops.flash_attn 已注册")
        # 列出可用的 op
        for name in dir(torch.ops.flash_attn):
            if not name.startswith('_'):
                print(f"  - {name}")
except:
    pass

print("\n" + "=" * 60)
print("5. pip cache 检查")  
print("=" * 60)

result3 = subprocess.run([sys.executable, "-m", "pip", "cache", "list", "flash_attn"], capture_output=True, text=True)
if result3.stdout.strip():
    print("pip 缓存中的 flash_attn wheels:")
    print(result3.stdout)
else:
    print("pip 缓存中没有 flash_attn（或 pip cache 命令不可用）")

# 也检查 flash-attn（连字符）
result4 = subprocess.run([sys.executable, "-m", "pip", "cache", "list", "flash-attn"], capture_output=True, text=True)
if result4.stdout.strip():
    print("pip 缓存中的 flash-attn wheels:")
    print(result4.stdout)

print("\n" + "=" * 60)
print("6. 编译兼容性验证")
print("=" * 60)

# 检查 flash_attn 编译时的 PyTorch/CUDA 版本 (如果有 metadata)
try:
    import importlib.metadata
    meta = importlib.metadata.metadata("flash-attn")
    print(f"Package metadata:")
    for key in ["Name", "Version", "Summary"]:
        val = meta.get(key)
        if val:
            print(f"  {key}: {val}")
    
    # 检查 wheel 文件名（如果能找到）
    dist = importlib.metadata.distribution("flash-attn")
    if hasattr(dist, '_path'):
        print(f"  dist path: {dist._path}")
except:
    pass

# 最关键的检查：torch 的 CUDA 版本和 flash_attn 编译时用的是否一致
print(f"\ntorch CUDA: {torch.version.cuda}")
print(f"torch ABI:  {torch._C._GLIBCXX_USE_CXX11_ABI}")

# 检查 flash_attn 的 _C 扩展是否能正常加载
print("\n尝试直接调用底层 CUDA 内核...")
try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward
    
    q = torch.randn(64, 48, 128, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn(64, 8, 128, device="cuda:0", dtype=torch.bfloat16)   # GQA: 8 kv heads
    v = torch.randn(64, 8, 128, device="cuda:0", dtype=torch.bfloat16)
    cu = torch.tensor([0, 64], dtype=torch.int32, device="cuda:0")
    
    print(f"  测试 GQA 配置: q_heads=48, kv_heads=8 (和 Qwen3.5-9B 一致)")
    out = _flash_attn_varlen_forward(q, k, v, cu, cu, 64, 64, 0.0, 1.0, True, None, None)
    torch.cuda.synchronize()
    print(f"  ✅ GQA 底层调用通过")
except Exception as e:
    print(f"  ❌ GQA 底层调用失败: {e}")
    print(f"  → 这说明 flash_attn 的 GQA 内核有问题!")

# 对比：非 GQA（和 Step 1 裸测一样，q/k/v 头数相同）
try:
    q2 = torch.randn(64, 48, 128, device="cuda:0", dtype=torch.bfloat16)
    k2 = torch.randn(64, 48, 128, device="cuda:0", dtype=torch.bfloat16)  # 同样 48 heads
    v2 = torch.randn(64, 48, 128, device="cuda:0", dtype=torch.bfloat16)
    
    print(f"  测试非 GQA 配置: q_heads=48, kv_heads=48")
    out2 = _flash_attn_varlen_forward(q2, k2, v2, cu, cu, 64, 64, 0.0, 1.0, True, None, None)
    torch.cuda.synchronize()
    print(f"  ✅ 非 GQA 底层调用通过")
except Exception as e:
    print(f"  ❌ 非 GQA 也失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)
print("\n如果 GQA 底层调用失败但非 GQA 通过，问题就是 flash_attn 2.8.3")
print("的 GQA 内核和当前 PyTorch/CUDA 不兼容。")
print("修复方案：从源码重新编译 flash_attn：")
print("  pip uninstall flash-attn -y")
print("  pip install flash-attn --no-build-isolation --no-cache-dir")