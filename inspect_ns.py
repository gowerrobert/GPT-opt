import torch
import matplotlib.pyplot as plt
from gptopt.linalg_utils import ns_pinv, ns_pinv_v2, rel_err, power_method, accelerated_ns_pinv

A = torch.load("/mnt/home/nghosh/GPT-opt/debug_matrices/C_transformer.h.0.attn.c_attn.weight_step_1.pt")
G = torch.load("/mnt/home/nghosh/GPT-opt/debug_matrices/grad_transformer.h.0.attn.c_attn.weight_step_1.pt")

print("A shape:", A.shape)
print("||A||_2", torch.linalg.norm(A, ord=2).item())
print("||A||_F: ", torch.linalg.norm(A, ord='fro').item())
print("sqrt ||A^2||_F", torch.linalg.norm(A @ A.T, ord='fro').sqrt().item())
print("sqrt(||A||_1 ||A||_inf)", (torch.linalg.norm(A, ord=1) * torch.linalg.norm(A, ord=float("inf"))).sqrt().item())

# Compute eigenvalues and sort in descending order
A = A.to(torch.float64)
eigvals, eigvecs = torch.linalg.eigh(A)
print(eigvals[:10])
print(eigvals[-10:])
print("symmetry error:", (A - A.T).abs().max().item())
print("eig error", (A - eigvecs @ torch.diag(eigvals) @ eigvecs.T).abs().max().item())
print("conditioning (before clipping)", f"{(eigvals.abs().max()/eigvals.abs().min()).item():.3e}")

plt.figure()
plt.plot(reversed(eigvals.cpu()))
plt.title("Eigenvalue spectrum")
plt.xlabel("Index")
plt.ylabel("Eigenvalue") 
plt.yscale("log")
plt.savefig("inspect_figs/spectrum.png")

clip = False
diag = False
if clip:  # clip negative eigenvalues
    eigvals = torch.maximum(eigvals, torch.tensor(1e-6))
if diag:
    A = torch.diag(eigvals).to(torch.float32)
else:
    A = (eigvecs @ torch.diag(eigvals) @ eigvecs.T).to(torch.float32)

dtype = torch.float32
l = 1e-3

power_norm = power_method(A, psd=True, use_bf16=False) #.item()
eps = power_norm * l
print("power_norm =", power_norm.item(), "\nfrobenius norm =", A.norm().item(), "\ntrue norm =", torch.linalg.norm(A, ord=2).item())
print(f"eps={eps}")

ns_A_pinv, ns_diagnostics = ns_pinv(A, max_steps=80, use_double=(dtype==torch.float64), use_bf16=(dtype==torch.float16), diagnostics=True)
ns2_A_pinv, ns2_diagnostics = ns_pinv_v2(A, eps=eps, max_steps=80, use_double=(dtype==torch.float64), use_bf16=(dtype==torch.float16), diagnostics=True)
ns_accel_A_pinv, ns_accel_diagnostics = accelerated_ns_pinv(A, eps, 1.1*power_norm, 80, psd=False, early_stop_eps=eps, dtype=dtype, diagnostics=True)
# _, cubic_diagnostics = accelerated_ns_pinv(A, l*power_norm, 1.1*power_norm, 80, psd="cubic", early_stop_eps=eigvals.abs().min(), dtype=dtype, diagnostics=True)
_, add_eps_diagnostics = accelerated_ns_pinv(A, eps, 1.1*power_norm, 80, psd=True, add_eps=1e-3, early_stop_eps=eps, dtype=dtype, diagnostics=True)

metric = 'AX'
end = None
plt.figure()
plt.plot([res[metric] for res in ns_diagnostics['residuals']][:end], label="NS", marker='.')
plt.plot([res[metric] for res in ns2_diagnostics['residuals']][:end], label="NS2", marker='.')
plt.plot([res[metric] for res in ns_accel_diagnostics['residuals']][:end], label="Accel", marker='.')
# plt.plot([res[metric] for res in cubic_diagnostics['residuals']][:end], label="Cubic", marker='.')
plt.plot([res[metric] for res in add_eps_diagnostics['residuals']][:end], label="Add eps", marker='.')
plt.title(f"Newton-Schulz iteration error ({metric})\neps=norm * {l:.1e}")
plt.xlabel("Iteration")
plt.ylabel(f"Relative Error: {metric}")
plt.yscale("log")
plt.legend()
plt.savefig("inspect_figs/convergence.png")
