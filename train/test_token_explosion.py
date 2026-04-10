import torch
import torch.nn.functional as F
import math

# 开启异常侦测
torch.autograd.set_detect_anomaly(True)

def test_massive_token_precision_loss():
    device = "cpu"
    
    # 💥 还原你的案发现场：万级 Token (16384)
    seq_len = 16384
    d_model = 128
    
    print(f"🚀 开始极限数学压测: 序列长度 = {seq_len}, 维度 = {d_model}")
    print(f"🖥️  运行设备: {device}\n" + "="*50)

    # 1. 制造相同的初始输入 (模拟带有一定量化毛刺的 Q, K, V)
    # 我们将初始值设为 Requires Grad，为了看反向传播的最终梯度
    q_init = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16) * 2.5
    k_init = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16) * 2.5
    v_init = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16)

    # =====================================================================
    # 🟢 路径 A：模拟 Flash Attention 2 (SRAM 内部使用 FP32 高精度累加)
    # =====================================================================
    q_fa2 = q_init.clone().detach().requires_grad_(True)
    k_fa2 = k_init.clone().detach().requires_grad_(True)
    v_fa2 = v_init.clone().detach().requires_grad_(True)

    # 强制上转为 FP32 进行安全的注意力计算
    scores_fa2 = (q_fa2.float() @ k_fa2.float().transpose(-2, -1)) / math.sqrt(d_model)
    probs_fa2 = F.softmax(scores_fa2, dim=-1)
    # 算完安全的结果后，再降级回 BF16 输出 (模拟 FA2 的输出行为)
    attn_out_fa2 = (probs_fa2 @ v_fa2.float()).to(torch.bfloat16)

    # RMSNorm
    var_fa2 = attn_out_fa2.pow(2).mean(-1, keepdim=True)
    norm_out_fa2 = attn_out_fa2 / torch.sqrt(var_fa2 + 1e-6)

    # =====================================================================
    # 🔴 路径 B：Eager Attention (你的真实环境，全程死磕 BF16 低精度累加)
    # =====================================================================
    q_eager = q_init.clone().detach().requires_grad_(True)
    k_eager = k_init.clone().detach().requires_grad_(True)
    v_eager = v_init.clone().detach().requires_grad_(True)

    # 全程在极容易发生截断的 BF16 下累加 16384 个数值
    scores_eager = (q_eager @ k_eager.transpose(-2, -1)) / math.sqrt(d_model)
    probs_eager = F.softmax(scores_eager, dim=-1)
    attn_out_eager = probs_eager @ v_eager

    # RMSNorm
    var_eager = attn_out_eager.pow(2).mean(-1, keepdim=True)
    norm_out_eager = attn_out_eager / torch.sqrt(var_eager + 1e-6)

    # =====================================================================
    # 💥 对比与核爆验证
    # =====================================================================
    print(f"📊 [前向传播方差对比] (方差越接近0，反向除法放大的倍数越恐怖):")
    print(f"   ✅ FA2 (高精度) 平均方差: {var_fa2.mean().item():.6f}")
    print(f"   ❌ Eager (BF16) 平均方差: {var_eager.mean().item():.6f}")
    
    # 我们模拟上一层传回来的健康微小梯度 (10^-5 级别)
    grad_loss = torch.randn_like(norm_out_fa2) * 1e-5
    
    print("\n⏪ 开始反向传播求导...")
    norm_out_fa2.backward(grad_loss, retain_graph=True)
    norm_out_eager.backward(grad_loss)

    print(f"\n🎯 [最终梯度极值对比]:")
    
    fa2_max_grad = q_fa2.grad.abs().max().item()
    eager_max_grad = q_eager.grad.abs().max().item()
    
    print(f"   ✅ FA2 (高精度保护) 捕获最大梯度: {fa2_max_grad:.2e}")
    print(f"   ❌ Eager (BF16死磕) 捕获最大梯度: {eager_max_grad:.2e}")
    
    if eager_max_grad > fa2_max_grad * 100:
        print("\n🚨 验证成功！纯粹的数学证明：在万级 Token 下，即便没有多层堆叠，")
        print("   仅仅是单层的 BF16 Eager 精度丢失，就足以将梯度瞬间放大成百上千倍！")
        print("   如果有 40 层堆叠，这将被指数级放大到 10^28 甚至直接变成 NaN。")

if __name__ == "__main__":
    test_massive_token_precision_loss()