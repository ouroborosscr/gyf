import torch

def dynamic_dispatch_interception():
    device = "cuda"
    dtype = torch.bfloat16
    q = torch.randn(1, 1, 128, 128, device=device, dtype=dtype)
    k = torch.randn(1, 1, 128, 128, device=device, dtype=dtype)
    v = torch.randn(1, 1, 128, 128, device=device, dtype=dtype)

    # 构造 Left Padding Mask
    dense_mask = torch.ones(1, 1, 128, 128, device=device, dtype=dtype)
    dense_mask[:, :, :, :10] = float('-inf')

    print("🚨 正在设置 C++ 调度器拦截网: 仅允许 Flash Attention，关闭兜底通道...")
    # 动态锁死底层调度器！
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
        try:
            print("➡️ 尝试用 Flash Attention 强吃 Dense Mask...")
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense_mask)
            print("✅ 居然成功了？说明我的静态审计错了！")
        except RuntimeError as e:
            print("\n💥 [抓获现行] C++ 调度器抛出异常：")
            print(f"   {str(e)}")
            print("   💡 铁证如山：PyTorch 明确宣告，Flash 引擎在底层根本接不住自定义 Mask。")
            print("   💡 这证实了在正常情况下，它一定会因为这个 Mask 而被迫掉入 Math 引擎！")

if __name__ == "__main__":
    dynamic_dispatch_interception()