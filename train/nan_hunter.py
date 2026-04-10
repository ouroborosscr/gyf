"""
NaN 猎手 v2 - 增加梯度钩子，精确定位 NaN 来源
确认是 backward 产生了 NaN 梯度，还是 optimizer.step 把正常梯度变成了 NaN 参数

用法同 v1
"""
import torch
import transformers
from datetime import datetime

class NaNHunterCallback(transformers.TrainerCallback):
    
    def __init__(self):
        self.step_count = 0
        self.hooks = []
        self.grad_snapshots = {}
        self.hooked = False
    
    def _install_hooks(self, model):
        if self.hooked:
            return
        
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_A" not in name and "lora_B" not in name:
                continue
            
            def make_hook(param_name):
                def hook(grad):
                    grad_nan = torch.isnan(grad).any().item()
                    grad_inf = torch.isinf(grad).any().item()
                    grad_max = grad.abs().max().item() if not (grad_nan or grad_inf) else float('inf')
                    grad_nan_count = torch.isnan(grad).sum().item() if grad_nan else 0
                    grad_inf_count = torch.isinf(grad).sum().item() if grad_inf else 0
                    
                    self.grad_snapshots[param_name] = {
                        "nan": grad_nan, "inf": grad_inf, "max": grad_max,
                        "nan_count": grad_nan_count, "inf_count": grad_inf_count,
                        "numel": grad.numel(),
                    }
                    
                    if grad_nan or grad_inf:
                        now = datetime.now().strftime("%H:%M:%S")
                        print(f"\n[🔍 梯度钩子 {now}] 🚨 {param_name}", flush=True)
                        print(f"  nan={grad_nan}({grad_nan_count}/{grad.numel()}) "
                              f"inf={grad_inf}({grad_inf_count}/{grad.numel()}) "
                              f"max={grad_max}", flush=True)
                    return grad
                return hook
            
            h = p.register_hook(make_hook(name))
            self.hooks.append(h)
        
        self.hooked = True
        print(f"[🔍 NaN猎手] 已在 {len(self.hooks)} 个 LoRA 参数上安装梯度钩子", flush=True)
    
    def _check_params(self, model):
        nan_p, inf_p, max_abs = [], [], 0.0
        for name, p in model.named_parameters():
            if p.data is None: continue
            d = p.data
            if torch.isnan(d).any():
                nan_p.append((name, torch.isnan(d).sum().item(), d.numel()))
            if torch.isinf(d).any():
                inf_p.append((name, torch.isinf(d).sum().item(), d.numel()))
            m = d.abs().max().item()
            if m > max_abs: max_abs = m
        return nan_p, inf_p, max_abs
    
    def _check_logits(self, model):
        device = next(model.parameters()).device
        try:
            with torch.no_grad():
                out = model(input_ids=torch.randint(0, 1000, (1, 32), device=device))
            logits = out.logits[0, -1, :]
            return {"nan": torch.isnan(logits).any().item(), "inf": torch.isinf(logits).any().item(),
                    "max": logits.abs().max().item() if not torch.isnan(logits).any() else float('nan')}
        except Exception as e:
            return {"error": str(e)[:80]}
    
    def _report(self, label, model):
        now = datetime.now().strftime("%H:%M:%S")
        nan_p, inf_p, max_abs = self._check_params(model)
        logits = self._check_logits(model)
        has_problem = len(nan_p) > 0 or len(inf_p) > 0 or logits.get("nan") or logits.get("inf") or "error" in logits
        icon = "❌" if has_problem else "✅"
        lstr = f"max={logits['max']:.2f}" if isinstance(logits.get('max'), float) and logits['max'] < 1e10 else "nan/inf"
        print(f"\n[🔍 {now}] {icon} {label}", flush=True)
        print(f"  参数: max={max_abs:.6f} nan={len(nan_p)} inf={len(inf_p)}", flush=True)
        print(f"  Logits: {lstr} nan={logits.get('nan','?')} inf={logits.get('inf','?')}", flush=True)
        if nan_p:
            for name, cnt, total in nan_p[:10]:
                print(f"  🚨 NaN: {name} ({cnt}/{total})", flush=True)
        if inf_p:
            for name, cnt, total in inf_p[:10]:
                print(f"  🚨 Inf: {name} ({cnt}/{total})", flush=True)
        return has_problem
    
    def _report_grads(self):
        if not self.grad_snapshots:
            return
        now = datetime.now().strftime("%H:%M:%S")
        nan_g = {k: v for k, v in self.grad_snapshots.items() if v["nan"]}
        inf_g = {k: v for k, v in self.grad_snapshots.items() if v["inf"]}
        
        if nan_g or inf_g:
            print(f"\n[🔍 {now}] 🚨 Step {self.step_count} backward 梯度异常!", flush=True)
            for name, info in list(nan_g.items())[:10]:
                print(f"  NaN梯度: {name} ({info['nan_count']}/{info['numel']})", flush=True)
            for name, info in list(inf_g.items())[:10]:
                print(f"  Inf梯度: {name} ({info['inf_count']}/{info['numel']})", flush=True)
            print(f"  → NaN 来自 BACKWARD", flush=True)
        else:
            max_name = max(self.grad_snapshots, key=lambda k: self.grad_snapshots[k]["max"] if self.grad_snapshots[k]["max"] != float('inf') else 0)
            max_val = self.grad_snapshots[max_name]["max"]
            print(f"\n[🔍 {now}] ✅ Step {self.step_count} 梯度全部正常 (最大={max_val:.6f} @ {max_name})", flush=True)
            print(f"  → 如果参数之后变NaN，则问题在 OPTIMIZER.STEP (DeepSpeed)", flush=True)
        
        self.grad_snapshots = {}
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._report("训练开始前", model)
            self._install_hooks(model)
    
    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self.step_count += 1
        if model is not None:
            self._report(f"Step {self.step_count} 开始前", model)
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._report_grads()
            has_problem = self._report(f"Step {self.step_count} 结束后", model)
            if has_problem:
                print(f"\n  💀 模型已损坏", flush=True)