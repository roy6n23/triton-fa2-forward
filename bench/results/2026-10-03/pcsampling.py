import sys, torch
sys.path.insert(0, "/root/triton-fa2-forward")
import triton.profiler as proton
from fa2.kernel import attention
q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
attention(q, k, v); torch.cuda.synchronize()
proton.start("/root/pcs", backend="cupti", mode="pcsampling")
attention(q, k, v); torch.cuda.synchronize()
proton.finalize()
print("pcsampling finished")
