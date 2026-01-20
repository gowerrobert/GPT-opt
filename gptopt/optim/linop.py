import torch
from torchmin.lstsq.linear_operator import TorchLinearOperator
from typing import Optional, Callable, Literal, Any
from torch._vmap_internals import _vmap 



    

def attn_linop_from_matrices(A1, A2):
    matvec = matvec_attn_linop(A1=A1, A2=A2)
    rmatvec = matvec_attn_linop_adj(A1=A1, A2=A2)
    fro_norm = fro_norm_attn_linop(A1=A1, A2=A2)
    m, n = A1.shape
    A_linop = TorchLinearOperator(matvec=matvec, rmatvec=rmatvec, 
                                    shape=(n**2, 2*m*n))
    A_linop.fro_norm = fro_norm
    A_linop.device = A1.device
    A_linop.A1 = A1
    A_linop.A2 = A2
    A_linop.dtype = A1.dtype
    return A_linop


def matvec_attn_linop(*, A1, A2): 
    # A(Z) = Z1^T A1 + Z2^T A2
    return lambda Z: mathcal_A_linop(A1=A1, A2=A2, Z=Z)

def matvec_attn_linop_adj(*, A1, A2): 
    # A^*(Y) = [A1 Y^T; A2 Y]
    return lambda Y: mathcal_A_adj_linop(A1=A1, A2=A2, Y=Y)

def fro_norm_attn_linop(*, A1, A2):
    return torch.sqrt(torch.norm(A1, p='fro')**2 + torch.norm(A2, p='fro')**2).item()

def mathcal_A_linop(*, A1, A2, Z): # A operator
    m = A1.shape[0]
    return mathcal_A_linop_base(A1=A1, A2=A2, Z1=Z[:m, :], Z2=Z[m:, :])


def mathcal_A_linop_base(*, A1, A2, Z1, Z2): # A operator
    return Z1.T @ A1 + A2.T @ Z2
       
        
def mathcal_A_adj_linop(*, A1, A2, Y):         # A^* operator  
    return torch.cat([A1 @ Y.T, A2 @ Y], dim=0)


def pd_residuals_max_ball_linop(A_linop, Y, Z, Grad, beta, mu=0):
    r1, r1_rel = proj_subgrad_l1(A_linop.matvec(Z), Y, beta=beta)

    r2 = torch.norm(Grad + mu * Z + A_linop.rmatvec(Y), p='fro').item()
    norm2 = torch.norm(Grad, p='fro').item()
    if norm2 < 1e-6: norm2 = 1.0
    r2_rel = r2 / norm2
    return r1, r1_rel, r2, r2_rel




def proj_subgrad_l1(AZ, Y, beta=1, y_zero_tol_abs=1e-7, y_zero_tol_rel=1e-12):
    # \min_S \beta\|AZ/\beta - S\|_F s.t. S \in \partial \|\vec(Y)\|_1 
    S = torch.sign(Y)
    # treat small values of Y as zeros
    y_tol = y_zero_tol_rel * Y.abs().max().item() + y_zero_tol_abs
    S[Y.abs() <= y_tol] = torch.clamp((AZ[Y.abs() <= y_tol]/beta), -1.0, 1.0)
    r = beta * torch.norm(AZ / beta - S, p='fro').item()
    # normalize by natural value of \|\beta S \|_F
    norm = beta * (S.numel()**0.5)
    return r, r / norm


def pd_residuals_infty_ball(A, B, Y, Z1, Z2, G1, G2, beta, mu=0):
    # KKT residuals 
    # h = I_{||.||_\max \leq beta}
    # 0 \in \partial h^*(Y) - \mathcal{A}(Z)   -- primal residual
    # 0 = G + \mu Z + \mathcal{A}^*(Y).        -- dual residual
    AZ = Z1.T @ B + A.T @ Z2 
    r1, r1_rel = proj_subgrad_l1(AZ, Y, beta=beta)
    r2_1 = (G1 + mu * Z1 + B @ Y.t()).pow(2).sum().sqrt().item()
    r2_2 = (G2 + mu * Z2 + A @ Y).pow(2).sum().sqrt().item()
    r2 = (r2_1**2 + r2_2**2)**0.5 
    # normalize r2 by the ||G1, G2||_F
    norm2 = (G1.pow(2).sum() + G2.pow(2).sum()).sqrt().item()
    if norm2 < 1e-6: norm2 = 1.0
    r2_rel = r2 / norm2 
    return r1, r1_rel, r2, r2_rel


def pd_residuals_max_ball(A1, A2, Y, Z, G1, G2, beta, mu=0):
    m = A1.shape[0]
    Z1, Z2 = Z[:m, :], Z[m:, :] 
    r1, r1_rel, r2, r2_rel = pd_residuals_infty_ball(B=A1, A=A2, Y=Y, Z1=Z1, Z2=Z2, G1=G1, G2=G2, 
                                   beta=beta, mu=mu)
    return r1, r1_rel, r2, r2_rel

