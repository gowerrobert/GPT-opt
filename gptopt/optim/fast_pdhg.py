import math
import torch
from .pdhg import *


"""
Adaptive restart and PID controller for PDHG primal weight
is based on cuPDLPx implementation: https://github.com/MIT-Lu-Lab/cuPDLPx
"""


def pdhg_initialize_variables(A1, Z0=None, Y0=None):
    m, n = A1.shape
    if Z0 is not None:
        Z = Z0 
    else:
        Z = torch.zeros((2 * m, n), device=A1.device, dtype=A1.dtype)
    Z_bar = Z.clone() 
    if Y0 is not None:
        Y = Y0
    else:
        Y = torch.zeros((n, n), device=A1.device, dtype=A1.dtype)
    return Z, Z_bar, Y


def pdhg_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t):
    if t < min_iter:
        return False 
    if (r1 < eps_abs and r2 < eps_abs) or (r1_rel < eps_rel and r2_rel < eps_rel):
        return True  
    return False


def check_adaptive_restart(
    solver_state,
    restart_params,
    termination_evaluation_frequency=100
):
    """
    Determine whether to perform an adaptive restart.
    """
    do_restart = False
    
    # CONDITION 1: First evaluation checkpoint - always restart
    if solver_state.total_count == termination_evaluation_frequency:
        do_restart = True
    
    elif solver_state.total_count > termination_evaluation_frequency: 
        
        # CONDITION 2: Sufficient reduction criterion
        # Error has dropped below 20% (default) of initial error
        if (solver_state.fixed_point_error <= 
            restart_params.sufficient_reduction_for_restart * solver_state.initial_fixed_point_error):
            do_restart = True
        
        # CONDITION 3: Necessary reduction + stagnation
        # Error is below 50% (default) of initial BUT is increasing
        if (solver_state.fixed_point_error <= 
            restart_params.necessary_reduction_for_restart *  solver_state.initial_fixed_point_error):
            
            if (solver_state.fixed_point_error > 
                solver_state.last_trial_fixed_point_error):
                do_restart = True
        
        # CONDITION 4: Artificial restart threshold
        # Too many iterations without restart (default:  36% of total iterations)
        if (solver_state.inner_count >= 
            restart_params.artificial_restart_threshold * solver_state.total_count):
            do_restart = True
    
    # Update for next check
    solver_state.last_trial_fixed_point_error = solver_state.fixed_point_error
    
    return do_restart


def perform_restart(state, restart_params):
    """
    Perform restart and update primal weight using PID controller.
    
    This matches the exact implementation in src/solver.cu lines 782-846.
    """
    
    # Compute delta solutions (change since last restart)
    # delta_primal = pdhg_primal - initial_primal
    # delta_dual = pdhg_dual - initial_dual
    
    # Compute L2 norms of the changes
    primal_dist = state.delta_primal_solution.pow(2).sum().sqrt().item()
    dual_dist = state.delta_dual_solution.pow(2).sum().sqrt().item()
    
    # Compute ratio of infeasibilities
    ratio_infeas = (state.relative_dual_residual / 
                   max(1e-8, state.relative_primal_residual))
    
    # Check if distances are in reasonable range for PID update
    if (primal_dist > 1e-16 and dual_dist > 1e-16 and 
        primal_dist < 1e12 and dual_dist < 1e12 and
        ratio_infeas > 1e-8 and ratio_infeas < 1e8):
        
        # ========================================
        # PID ERROR DEFINITION (THE KEY PART!)
        # ========================================
        error = (np.log(dual_dist) - 
                 np.log(primal_dist) - 
                 np.log(state.primal_weight))
        
        # Update integral term with smoothing
        # New integral = i_smooth × old_integral + error
        state.primal_weight_error_sum *= restart_params.i_smooth
        state.primal_weight_error_sum += error
        
        # Compute derivative term (change in error)
        delta_error = error - state.primal_weight_last_error
        
        # PID control update
        pid_update = (restart_params.k_p * error +
                     restart_params.k_i * state.primal_weight_error_sum +
                     restart_params.k_d * delta_error)
        
        # Update primal weight (exponential for positivity)
        state.primal_weight *= np.exp(pid_update)
        
        # Save error for next iteration
        state.primal_weight_last_error = error
    else:
        # Fallback:  use best known primal weight and reset PID state
        state.primal_weight = state.best_primal_weight
        state.primal_weight_error_sum = 0.0
        state.primal_weight_last_error = 0.0
    
    # Track best primal weight based on residual balance
    primal_dual_residual_gap = abs(
        np.log10(state.relative_dual_residual / max(1e-8, state.relative_primal_residual))
    )
    if primal_dual_residual_gap < state.best_primal_dual_residual_gap:
        state.best_primal_dual_residual_gap = primal_dual_residual_gap
        state.best_primal_weight = state.primal_weight
    
    # Copy current solutions to initial (restart averaging)
    state.initial_primal_solution = state.pdhg_primal_solution.clone() 
    state.initial_dual_solution = state.pdhg_dual_solution.clone() 
    
    # Reset inner counter
    state.inner_count = 0
    state.last_trial_fixed_point_error = np.inf


class SolverState:
    """Extended solver state with all necessary fields."""
    def __init__(self, Z: torch.Tensor, Y: torch.Tensor, A1: torch.Tensor, A2: torch.Tensor):
        # Iteration counters
        self.total_count = 0
        self.inner_count = 0
        self.A1 = A1
        self.A2 = A2
        
        self.pdhg_primal_solution = Z
        self.pdhg_dual_solution = Y
        self.initial_primal_solution = Z.clone()
        self.initial_dual_solution = Y.clone() 

        self.delta_primal_solution = torch.zeros_like(Z)
        self.delta_dual_solution = torch.zeros_like(Y)
        
        # Residuals
        self.relative_primal_residual = 1.0
        self.relative_dual_residual = 1.0
        
        # Fixed point error tracking
        self.fixed_point_error = np.inf
        self.initial_fixed_point_error = np.inf
        self.last_trial_fixed_point_error = np.inf
        
        # PID controller state
        self.primal_weight = 1.0
        self.primal_weight_error_sum = 0.0
        self.primal_weight_last_error = 0.0
        self.best_primal_weight = 1.0
        self.best_primal_dual_residual_gap = np.inf


    def update(self, Z_new: torch.Tensor, Y_new: torch.Tensor, rho, gamma):
        self.pdhg_primal_solution = Z_new
        self.pdhg_dual_solution = Y_new
        self.total_count += 1
        self.inner_count += 1

        self.pdhg_primal_solution = Z_new
        self.pdhg_dual_solution = Y_new 

        self.delta_primal_solution = Z_new - self.initial_primal_solution
        self.delta_dual_solution = Y_new - self.initial_dual_solution

        if self.initial_fixed_point_error==np.inf or self.inner_count == 1:
            self.initial_fixed_point_error = self.fixed_point_error


class RestartParams:
    """Restart parameters matching cuPDLPx defaults."""
    def __init__(self):
        self.artificial_restart_threshold = 0.36
        self.sufficient_reduction_for_restart = 0.2
        self.necessary_reduction_for_restart = 0.5
        self.k_p = 0.99   # Proportional gain
        self.k_i = 0.01   # Integral gain
        self.k_d = 0.0    # Derivative gain
        self.i_smooth = 0.3  # Integral smoothing


def pdhg_kq_attn_layer(
    prox_h_conj,
    A1: torch.Tensor,
    A2: torch.Tensor,
    G1: torch.Tensor,
    G2: torch.Tensor,
    beta: float,
    mu: float = 0.0,
    lamb_max=None,
    max_iter: int = 100,
    eps_abs: float = 1e-3,
    eps_rel: float = 1e-3,
    stopping: bool = False,
    min_iter: int = 10,
    Y0: torch.Tensor | None = None,  
    Z0: torch.Tensor | None = None, 
    diag_scaling: bool = False,
    h_conj= None,
    f_star: float | None = None,
    pd_residuals: Optional[callable] = None, 
    verbose: bool = False,
    equilibration: bool = False,
    reflected_halpern: bool = False,
    enable_restart: bool = False, \
    theta=1.0 
    ): 
    """
    PDHG method for solving 
                minimize_Z      tr(G^TZ) + h(mathcal_A(Z)) + (mu/2) ||Z||_F^2
    """
    if lamb_max is None:
        nA = A1.pow(2).sum().sqrt().item()
        nB = A2.pow(2).sum().sqrt().item()
        lamb_max = (nA * nA + nB * nB) ** 0.5  

    Z, Z_bar, Y = pdhg_initialize_variables(A1=A1, Z0=Z0, Y0=Y0) 
    Grad = torch.cat([G1, G2], dim=0)
    m, n = A1.shape

    record = PDHGResidualRecorder(pd_residuals=pd_residuals,
                                    A1=A1, A2=A2, G1=G1, G2=G2,
                                    beta=beta, mu=mu, f_star=f_star)
    base_step = max(min(0.998 / (lamb_max + 1e-12), 1e4), 1e-5)
    if diag_scaling:
        R, Gamma1, Gamma2 = pdhg_diagonal_scaling(A=A2, B=A1, eta=0.99)
        Gamma = torch.cat([Gamma1, Gamma2], dim=0)
        base_step = 1
    elif equilibration:
        R, Gamma1, Gamma2 = ruiz_equilibration(A1=A1, A2=A2, num_iters=50, eps=1e-8)
        Gamma = torch.cat([Gamma1, Gamma2], dim=0).pow(2)
        R = R.pow(2)
        lamb_max = R.max() * (nA**2 * Gamma1.max()**4 + nB**2 * Gamma2.max()**4 ) ** 0.5 
        base_step = max(min(0.998 / (lamb_max + 1e-12), 1e4), 1e-5)
    else:
        R = torch.ones((n, n), device=Y.device, dtype=Y.dtype)
        Gamma = torch.ones((2 * m, 1), device=A1.device, dtype=A1.dtype) 

    dual_val = None 
    if verbose:
        print(f"||A||_op <= {lamb_max:.4e}")

    
    record.record(0, Y=Y, Z=Z, dual_val=dual_val)
    state = SolverState(Z=Z, Y=Y, A1=A1, A2=A2)
    if enable_restart:
        restart_params = RestartParams()

    best_res = np.inf

    for t in range(max_iter):
        if enable_restart: 
            gamma = base_step * state.primal_weight**0.5
            rho   = base_step / state.primal_weight**0.5
        else: 
            gamma = rho = base_step

        # PDHG updates
        U = Y + rho * R * mathcal_A_linop(A1=A1, A2=A2, Z=Z_bar)
        Y_new = prox_h_conj(U, rho, R=R)
        Z_new = (1 / (1 + gamma * Gamma * mu)) * (Z - gamma * Gamma * (mathcal_A_adj_linop(A1=A1, A2=A2, Y=Y_new) + Grad))

        state.relative_primal_residual   = ((mathcal_A_linop(A1=A1, A2=A2, Z=Z_new) - (U - Y_new) * (1 / (rho * R))).pow(2).sum().sqrt() / (max(1e-8, 1e-4 * Z.numel() + U.pow(2).sum().sqrt()))).item()
        del U
        state.fixed_point_error = ((1 / gamma) * ((Z - Z_new) / Gamma).pow(2).sum() \
                                    + 2 * ((Y - Y_new) * mathcal_A_linop(A1=A1, A2=A2, Z=(Z - Z_new))).sum() \
                                    + (1 / rho) * ((Y - Y_new) / R).pow(2).sum()).pow(0.5).item()
        
        # reflected halpern
        if reflected_halpern: 
            if enable_restart:
                k = state.inner_count
            else:
                k = t
            Y_new = ((k + 1)/ (k + 2)) * (2 * Y_new - Y)  + (1 / (k + 2)) * state.initial_dual_solution
            Z_new = ((k + 1) / (k + 2)) * (2 * Z_new - Z) + (1 / (k + 2)) * state.initial_primal_solution

        # if primal is strongly convex -- record dual values
        if mu > 0 and h_conj is not None: 
            dual_val = - h_conj(Y_new) - (1/(2 * mu)) * (A1 @ Y_new.t() + G1).pow(2).sum()
            dual_val = dual_val - (1/(2 * mu)) * (A2 @ Y_new + G2).pow(2).sum() 

        r1, r1_rel, r2, r2_rel = record.record(t, Y=Y_new, Z=Z_new, dual_val=dual_val)

        state.relative_dual_residual = r2_rel

        #  adaptive restart
        Z_bar = Z_new + theta * (Z_new - Z)
        if enable_restart:
            state.update(Z_new=Z_new, Y_new=Y_new, rho=rho, gamma=gamma)

            if check_adaptive_restart(state, restart_params):
                perform_restart(state, restart_params)
                # reset anchor
                Z_bar = Z_new 
                if verbose:
                    print(f"[RESTART] iter={t:4d} primal_weight={state.primal_weight:.4e} "
                          f"r_prim={r1_rel:.3e} r_dual={r2_rel:.3e}")
        
        Z = Z_new
        Y = Y_new
        if best_res > r1_rel + r2_rel:
            best_res = r1_rel + r2_rel
            best_Z = Z.clone()
            best_Y = Y.clone()

        if pdhg_stopping_criteria(r1, r2, r1_rel, r2_rel, eps_abs, eps_rel, min_iter, t): 
            break 

    return best_Z, record.as_dict(), (best_Y.pow(2).sum()).sqrt().item(), best_Y


def ruiz_equilibration(A1: torch.Tensor, A2: torch.Tensor, num_iters=10, eps=1e-8, debug=False):
    """
    Ruiz equilibration for linear operator 
        \mathcal{A}(Z) = Z1^T A1 + A2^T Z2
    returns R, Gamma1, Gamma2 such that the matrix \tilde K of the the equilibrated operator
        \tilde \mathcal{A}(Z) = R * (Z1^T (Gamma2 A2) + (Gamma1 A1)^T Z2)
    is s.t. -1 <= \tilde K_{ij} <= 1
    where \vec{\tilde \mathcal{A}(Z)} = \tilde K [\vec{Z1}; \vec{Z2}]
    """
    device, dtype = A1.device, A1.dtype
    p1, n = A1.shape
    p2, n2 = A2.shape
    assert n == n2

    R = torch.ones((n, n), device=device, dtype=dtype)
    Gamma1 = torch.ones((p1, n), device=device, dtype=dtype)
    Gamma2 = torch.ones((p2, n), device=device, dtype=dtype)

    absA1 = A1.abs()
    absA2 = A2.abs()

    def inv_sqrt_pos(x: torch.Tensor) -> torch.Tensor:
        # 1/sqrt(x) for x>eps else 1 (do nothing if identically zero)
        return torch.where(x > eps, torch.rsqrt(x), torch.ones_like(x))

    for _ in range(num_iters): 
        # ---- Row max for K rows (i,j) ----
        # term1(i,j) = max_ℓ |Gamma1_{ℓ,i}| |A1_{ℓ,j}|
        term1 = (Gamma1[:, :, None] * absA1[:, None, :]).amax(dim=0)          # (n,n)
        # term2(i,j) = max_ℓ |A2_{ℓ,i}| |Gamma2_{ℓ,j}|
        term2 = (absA2[:, :, None] * Gamma2[:, None, :]).amax(dim=0)          # (n,n)
        row_max = R * torch.maximum(term1, term2)                         # (n,n)

        # ---- Column max for Z1 columns (ℓ,i) ----
        # m1(ℓ,i) = max_j |A1_{ℓ,j}| |R_{i,j}|
        m1 = (absA1[:, None, :] * R[None, :, :]).amax(dim=2)              # (p1,n)
        col1_max = Gamma1 * m1
        

        # ---- Column max for Z2 columns (ℓ,j) ----
        # m2(ℓ,j) = max_i |A2_{ℓ,i}| |R_{i,j}|
        m2 = (absA2[:, :, None] * R[None, :, :]).amax(dim=1)              # (p2,n)
        col2_max = Gamma2 * m2

        # ---- Update ----
        Gamma1 = Gamma1 * inv_sqrt_pos(col1_max)
        Gamma2 = Gamma2 * inv_sqrt_pos(col2_max)
        R = R * inv_sqrt_pos(row_max)

    return R, Gamma1, Gamma2


