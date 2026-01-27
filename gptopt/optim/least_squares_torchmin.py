import torch
from torchmin.lstsq.linear_operator import TorchLinearOperator 
from torchmin.lstsq.lsmr import lsmr

from .linop import *







def solve_lsmr_Y_lstsq(A_linop, Grad, n_head=1, maxiter=1000, atol=1e-6, btol=1e-6):
    # Minimize $\|\mathcal{A}^*(Y) + G\|_F^2$
    m, n = linop_mn_from_shape(A_linop, n_head=n_head)

    A_linop_vec = wrap_Astar_for_lsmr(A_linop, n_head=n_head)
    b_vec = pack_Z(-Grad, m, n, n_head=n_head)         
    y_vec, itn = lsmr(A_linop_vec, b_vec, maxiter=maxiter, atol=atol, btol=btol)    
    Y_hat = unpack_Y(y_vec, n, n_head=n_head)
    res = torch.norm(A_linop.rmatvec(Y_hat) + Grad, p='fro') / torch.norm(Grad, p='fro')
    return Y_hat, res, itn


def solve_lsmr_Z_lstsq(A_linop, beta, Y0, n_head=1, maxiter=1000, atol=1e-6, btol=1e-6):
    # Minimize $\|\mathcal{A}(Z) +\beta \mathbf{sign}(Y^0)\|_F^2$
    m, n = linop_mn_from_shape(A_linop, n_head=n_head)

    A_linop_vec = wrap_A_for_lsmr(A_linop, n_head=n_head)
    S = torch.sign(Y0)
    b_vec = pack_Y(-beta * S, n, n_head=n_head)         
    z_vec, itn = lsmr(A_linop_vec, b_vec, maxiter=maxiter, atol=atol, btol=btol)    
    Z_hat = unpack_Z(z_vec, m, n, n_head=n_head)
    res = torch.norm(A_linop.matvec(Z_hat) + beta * S, p='fro') / torch.norm(beta * S, p='fro')
    return Z_hat, res, itn



def wrap_Astar_for_lsmr(A_linop, n_head):
    m, n = linop_mn_from_shape(A_linop, n_head=n_head)

    def mv(y_vec):
        Y = unpack_Y(y_vec, n, n_head=n_head)
        Z = A_linop.rmatvec(Y)                        # A^*(Y), shape (2hm, n)
        return pack_Z(Z, m, n, n_head=n_head)         # length 2hmn

    def rmv(z_vec):
        Z = unpack_Z(z_vec, m, n, n_head=n_head)      # shape (2hm, n)
        Y = A_linop.matvec(Z)                         # A(Z), shape (hn, n)
        return pack_Y(Y, n, n_head=n_head)            # length hn^2

    return TorchLinearOperator(shape=(2*n_head*m*n, n_head*n*n), matvec=mv, rmatvec=rmv)


def wrap_A_for_lsmr(A_linop, n_head):
    m, n = linop_mn_from_shape(A_linop, n_head=n_head)

    def rmv(y_vec):
        Y = unpack_Y(y_vec, n, n_head=n_head)
        Z = A_linop.rmatvec(Y)                        # A^*(Y), shape (2hm, n)
        return pack_Z(Z, m, n, n_head=n_head)         # length 2hmn

    def mv(z_vec):
        Z = unpack_Z(z_vec, m, n, n_head=n_head)      # shape (2hm, n)
        Y = A_linop.matvec(Z)                         # A(Z), shape (hn, n)
        return pack_Y(Y, n, n_head=n_head)            # length hn^2

    return TorchLinearOperator(shape=(n_head*n*n, 2*n_head*m*n), matvec=mv, rmatvec=rmv)


