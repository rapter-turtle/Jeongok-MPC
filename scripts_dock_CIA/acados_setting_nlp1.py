from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt, log, exp, tanh

def export_heron_model_nlp1() -> AcadosModel:
    model_name = 'heron_nlp1'
    # constants

    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]
 
    # Mid speed
    Xu_dot = 15.26
    Xu = 1.671
    Xuu = 0.481
    # Slow speed
    # Xu = 0.783
    # Xuu = 2.22

    Yv = 0.1074
    Yvv = 0.0
    Yr = 0.0
    Nr = 0.3478
    Nrr = 0.3
    Nv = 0.0
    bu=  2.1/500.0
    b2 = 0.045/500.0
    b3 = 0.574
    F_bow = 0.02
    F_l = 3.0
    tau = 1.2

    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    u    = SX.sym('u')
    v    = SX.sym('v')
    r    = SX.sym('r')

    delta  = SX.sym('delta')
    F  = SX.sym('F')
    B  = SX.sym('B')
    F_eff = SX.sym('F_eff')

    states = vertcat(xn, yn, psi, u, v, r, delta, F_eff, F)

    delta_d  = SX.sym('delta_d')
    F_d  = SX.sym('F_d')
    inputs  = vertcat(delta_d, F_d, B)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    delta_dot   = SX.sym('delta_dot')
    F_dot   = SX.sym('F_dot')
    F_eff_dot = SX.sym('F_eff_dot')

    # set up parameters
    ox1 = SX.sym('ox1') 
    oy1 = SX.sym('oy1') 
    or1 = SX.sym('or1') 



    p = vertcat(ox1, oy1, or1)
    
    
    states_dot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, delta_dot, F_eff_dot, F_dot)


    # Deadzone
    s = 25
    k = 8
    a1 = 5.0
    a2 = 5.0
    # a1 = 2.2*2.2
    # a2 = 2.2*2.2
    b11 = 1.0
    b22 = 1.0

    eps = 0.00001
    # dynamics
    T = ((1/(1+exp(s*F_eff)))*(b11*F_eff + tanh(k*F_eff)*a1) + (1/(1+exp(-s*F_eff)))*(b22*F_eff + tanh(k*F_eff)*a2))

    
    f_expl = vertcat(u*cos(psi) - v*sin(psi),
                     u*sin(psi) + v*cos(psi),
                     r,
                     ( - Xu*u - Xuu * sqrt(u * u + eps) * u + T*cos(bu*delta))/(M + Xu_dot),
                     ( -Yv*v - Yr*r + T*sin(b2*delta) + F_bow*B),
                     ( - Nr*r - b3*T*sin(b2*delta) + F_l*F_bow*B),
                     delta_d,
                     (F - F_eff)/tau,
                     F_d
                     )

    f_impl = states_dot - f_expl


    num_obs = 2


    #docking
    xh = xn + 4*cos(psi)
    xb = xn - 1*cos(psi)
    yh = yn + 4*sin(psi)
    yb = yn - 1*sin(psi)

    xh_rot = xh*cos(or1) - yh*sin(or1)
    yh_rot = xh*sin(or1) + yh*cos(or1)
    
    xb_rot = xh*cos(or1) - yh*sin(or1)
    yb_rot = xh*sin(or1) + yh*cos(or1)
    
    dock_end_x = ox1*cos(or1) - oy1*sin(or1)
    dock_end_y = ox1*sin(or1) + oy1*cos(or1)

    h_expr = SX.zeros(num_obs,1)
    h_expr[0] = -yh_rot + dock_end_y + 1
    h_expr[1] = -yb_rot + dock_end_y + 1
    
    model = AcadosModel()
    model.con_h_expr = h_expr
    model.p = p 
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = states
    model.xdot = states_dot
    model.u = inputs
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]', '$y$ [m]',  '$psi$ [rad]',  '$u$ [m/s]', '$v$ [m/s]', '$r$ [rad/s]', '$delta$ [N]', '$F$ [N]']
    model.u_labels = ['$n_1_d$ [N/s]', '$n_2_d$ [N/s]', '$Bow$ [N]']
    model.t_label = '$t$ [s]'

    return model


def setup_trajectory_tracking_nlp1(x0, N_horizon, Tf, Q_mat, Q_mat_terminal, R_mat):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_heron_model_nlp1()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'


    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat_terminal

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))


    ocp.constraints.x0 = x0


    ocp.parameter_values = np.array([0.0, 0.0, 0.0])

    num_obs = 2
    ocp.constraints.uh = 1e10 * np.ones(num_obs)
    ocp.constraints.lh = np.zeros(num_obs)

    ocp.constraints.idxsh = np.array([0,1])
    ocp.constraints.idxsh_e = np.array([0,1])
    Zh = 1e4 * np.ones(num_obs)
    zh = 1e4 * np.ones(num_obs)
    ocp.cost.zl = zh
    ocp.cost.zu = zh
    ocp.cost.Zl = Zh
    ocp.cost.Zu = Zh
    ocp.cost.zl_e = zh
    ocp.cost.zu_e = zh
    ocp.cost.Zl_e = Zh
    ocp.cost.Zu_e = Zh

    # copy for terminal
    ocp.constraints.uh_e = ocp.constraints.uh
    ocp.constraints.lh_e = ocp.constraints.lh
    ocp.model.con_h_expr_e = ocp.model.con_h_expr

    # set constraints
    ocp.constraints.lbu = np.array([-100.0,-3, -1])
    ocp.constraints.ubu = np.array([+100.0,+3, +1])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    ocp.constraints.lbx = np.array([-250.0, -10.0])
    ocp.constraints.ubx = np.array([250.0, 16.0])
    ocp.constraints.idxbx = np.array([6, 8])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'#'ERK'
    ocp.solver_options.sim_method_newton_iter = 20
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_cond_N = N_horizon
    

    # set prediction horizon
    ocp.solver_options.tf = Tf
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
    # create an integrator with the same settings as used in the OCP solver.
    # acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver


