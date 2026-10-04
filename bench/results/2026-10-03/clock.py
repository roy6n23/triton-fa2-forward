import subprocess, sys, time, torch
sys.path.insert(0, "/root/triton-fa2-forward")
from fa2.kernel import attention
q = torch.randn(1, 32, 16384, 128, device="cuda", dtype=torch.float16)
k = torch.randn(1, 8, 16384, 128, device="cuda", dtype=torch.float16); v = torch.randn_like(k)
attention(q, k, v); torch.cuda.synchronize()
smi = subprocess.Popen(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active",
                        "--format=csv,noheader,nounits", "-lms", "200"], stdout=open("/root/clock.csv", "w"))
t0 = time.time(); n = 0
while time.time() - t0 < 15:
    for _ in range(20):
        attention(q, k, v)
    torch.cuda.synchronize(); n += 20
smi.terminate()
print(f"{n} launches in {time.time() - t0:.1f} s")
