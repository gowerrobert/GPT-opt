import torch
from torchmin.lstsq.linear_operator import TorchLinearOperator as TMLinearOperator
from torchmin.lstsq.lsmr import lsmr

from .linop import *



def solve_lsmr_Y_lstsq(A_linop, Grad):
    # Minimize $\|\mathcal{A}^*(Y) + G\|_F^2$
    n_sq, mn2 = A_linop.shape          # (n^2, 2mn)
    n = int(round(n_sq ** 0.5)) 
    m = mn2 // (2 * n)
    assert n * n == n_sq and 2 * m * n == mn2

    A_linop_vec = wrap_Astar_for_lsmr(A_linop)
    b_vec = pack_Z(-Grad, m, n)         
    y_vec, itn = lsmr(A_linop_vec, b_vec)    
    Y_hat = unpack_Y(y_vec, n)
    res = torch.norm(A_linop.rmatvec(Y_hat) + Grad, p='fro')/torch.norm(Grad, p='fro')
    return Y_hat, res, itn


def solve_lsmr_Z_lstsq(A_linop, beta, Y0):
    # Minimize $\|\mathcal{A}(Z) +\beta \mathbf{sign}(Y^0)\|_F^2$
    n_sq, mn2 = A_linop.shape          # (n^2, 2mn)
    n = int(round(n_sq ** 0.5)) 
    m = mn2 // (2 * n)
    assert n * n == n_sq and 2 * m * n == mn2

    A_linop_vec = wrap_A_for_lsmr(A_linop)
    S = torch.sign(Y0)
    b_vec = pack_Y(-beta * S, n)         
    z_vec, itn = lsmr(A_linop_vec, b_vec)    
    Z_hat = unpack_Z(z_vec, m, n)
    res = torch.norm(A_linop.matvec(Z_hat) + beta * S, p='fro')/torch.norm(beta * S, p='fro')
    return Z_hat, res, itn



def pack_Z(Z, m, n):
    # vec_Z(Z) = [vec(Z1); vec(Z2^T)]
    Z1, Z2 = Z[:m, :], Z[m:, :]
    return torch.cat([Z1.reshape(-1), Z2.T.reshape(-1)], dim=0)

def unpack_Z(z_vec, m, n):
    # inverse of pack_Z
    z_vec = z_vec.reshape(-1)
    Z1 = z_vec[:m*n].reshape(m, n)
    Z2 = z_vec[m*n:].reshape(n, m).T   # <-- critical
    return torch.cat([Z1, Z2], dim=0)  # (2m, n)

def pack_Y(Y, n):
    # vec_Y(Y) = vec(Y^T)
    return Y.T.reshape(-1)

def unpack_Y(y_vec, n):
    # inverse of pack_Y
    return y_vec.reshape(n, n).T


def wrap_Astar_for_lsmr(A_linop):
    n_sq, mn2 = A_linop.shape          # (n^2, 2mn)
    n = int(round(n_sq ** 0.5))
    assert n * n == n_sq
    m = mn2 // (2 * n)
    assert 2 * m * n == mn2

    def mv(y_vec):
        Y = unpack_Y(y_vec, n)
        Z = A_linop.rmatvec(Y)         # A^*(Y), shape (2m, n)
        return pack_Z(Z, m, n)         # length 2mn

    def rmv(z_vec):
        Z = unpack_Z(z_vec, m, n)      # shape (2m, n)
        Y = A_linop.matvec(Z)          # A(Z), shape (n, n)
        return pack_Y(Y, n)            # length n^2

    return TMLinearOperator(shape=(2*m*n, n*n), matvec=mv, rmatvec=rmv)


def wrap_A_for_lsmr(A_linop):
    n_sq, mn2 = A_linop.shape          # (n^2, 2mn)
    n = int(round(n_sq ** 0.5))
    assert n * n == n_sq
    m = mn2 // (2 * n)
    assert 2 * m * n == mn2

    def rmv(y_vec):
        Y = unpack_Y(y_vec, n)
        Z = A_linop.rmatvec(Y)         # A^*(Y), shape (2m, n)
        return pack_Z(Z, m, n)         # length 2mn

    def mv(z_vec):
        Z = unpack_Z(z_vec, m, n)      # shape (2m, n)
        Y = A_linop.matvec(Z)          # A(Z), shape (n, n)
        return pack_Y(Y, n)            # length n^2

    return TMLinearOperator(shape=(n*n, 2*m*n), matvec=mv, rmatvec=rmv)
