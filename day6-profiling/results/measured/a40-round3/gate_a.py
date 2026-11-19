import torch
x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(3):
    y = x @ x
torch.cuda.synchronize()
print("ok", y.float().mean().item())
