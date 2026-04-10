"""
在 GRPOTrainer 真实运行中拦截 loss
用法：trainer 创建后加：
    import loss_inspector
    loss_inspector.install(trainer)
"""
import torch
import logging

_original_compute_loss = None
_step_count = 0

def _patched_compute_loss(self, model, inputs):
    global _step_count
    _step_count += 1
    
    loss = _original_compute_loss(self, model, inputs)
    
    if _step_count <= 12:
        loss_val = loss.item() if not torch.isnan(loss) else "NaN"
        logging.info(
            f"🔬 [LossInspector] micro_step={_step_count}: "
            f"loss={loss_val}, dtype={loss.dtype}"
        )
        
        advantages = inputs.get("advantages")
        if advantages is not None:
            logging.info(
                f"   advantages: min={advantages.min().item():.4f}, "
                f"max={advantages.max().item():.4f}"
            )
        
        if loss.requires_grad:
            def backward_hook(grad):
                gval = grad.item() if grad.numel() == 1 else grad.abs().max().item()
                logging.info(f"🔬 [LossInspector] backward hook: grad={gval}, nan={torch.isnan(grad).any().item()}")
                return grad
            loss.register_hook(backward_hook)
    
    return loss

def install(trainer):
    global _original_compute_loss
    _original_compute_loss = trainer._compute_loss.__func__
    import types
    trainer._compute_loss = types.MethodType(_patched_compute_loss, trainer)
    logging.info("🔬 [LossInspector] 已安装")