from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt, log, exp, tanh

def export_heron_model() -> AcadosModel:
    model_name = 'heron'
    # constants

    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]

    Xu_dot = 1.74
    Xu = 1.671
    Xuu = 0.481
    Yv = 0.1074
    Yvv = 0.0
    Yr = 0.0
    Nr = 0.3478
    Nrr = 0.3
    Nv = 0.0
    bu=  1.74/500.0
    b2 = 0.045/500.0
    b3 = 0.574

    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    u    = SX.sym('u')
    v    = SX.sym('v')
    r    = SX.sym('r')

    delta  = SX.sym('delta')
    F_cmd  = SX.sym('F_cmd')

    states = vertcat(xn, yn, psi, u, v, r, delta, F_cmd)

    delta_d  = SX.sym('delta_d')
    F_cmd_d  = SX.sym('F_cmd_d')
    inputs  = vertcat(delta_d, F_cmd_d)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    delta_dot   = SX.sym('delta_dot')
    F_cmd_dot   = SX.sym('F_cmd_dot')

    # set up parameters
    ox1 = SX.sym('ox1') 
    oy1 = SX.sym('oy1') 
    or1 = SX.sym('or1') 
    ox2 = SX.sym('ox2') 
    oy2 = SX.sym('oy2') 
    or2 = SX.sym('or2') 


    du = SX.sym('du')
    dv = SX.sym('dv')
    dr = SX.sym('dr') 
    
    # du = dx*cos(psi) + dy*sin(psi)
    # dv = -dx*sin(psi) + dy*cos(psi)

    p = vertcat(ox1, oy1, or1, 
                ox2, oy2, or2,
                du, dv, dr)
    
    
    states_dot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, delta_dot, F_cmd_dot)

    # Deadzone
    s = 50
    k = 5
    a1 = 2.2*2.2
    a2 = 2.2*2.2
    b1 = 0.95
    b2 = 0.95

    F = F_cmd 
    # (1/(1+exp(s*F_cmd)))*(b1*F_cmd + tanh(k*F_cmd)*a1) + (1/(1+exp(-s*F_cmd)))*(b2*F_cmd + tanh(k*F_cmd)*a2)


    # dynamics
    eps = 0.00001
    f_expl = vertcat(u*cos(psi) - v*sin(psi),
                     u*sin(psi) + v*cos(psi),
                     r,
                     ( - Xu*u - Xuu * sqrt(u * u + eps) * u + F*cos(bu*delta))/(M + Xu_dot) - du,
                     ( -Yv*v - Yr*r + F*sin(b2*delta)) - dv,
                     ( - Nr*r - b3*F*sin(b2*delta)) - dr,
                     delta_d,
                     F_cmd_d
                     )

    f_impl = states_dot - f_expl


    num_obs = 3

    #docking
    xh_dot = u*cos(psi) - v*sin(psi) - r*4.0*sin(psi)
    yh_dot = u*sin(psi) + v*cos(psi) + r*4.0*cos(psi)
    xh = xn + 4.0*cos(psi)
    yh = yn + 4.0*sin(psi)
    # xh_dot = u*cos(psi) - v*sin(psi)
    # yh_dot = u*sin(psi) + v*cos(psi)
    # xh = xn
    # yh = yn
    
    # h_dock1 = 2.0 + ox2*exp(-0.1*(xh - ox1)) - (yh - oy1)
    # h_dock2 = 2.0 + ox2*exp(-0.1*(xh - ox1)) + (yh - oy1)
    # h_dock1_dot = -xh_dot*ox2*0.1*exp(-0.1*(xh - ox1)) + yh_dot
    # h_dock2_dot = -xh_dot*ox2*0.1*exp(-0.1*(xh - ox1)) - yh_dot


    h_expr = SX.zeros(num_obs,1)
    h_expr[0] = 2.0#h_dock1_dot + 10.0*h_dock1
    h_expr[1] = 2.0#h_dock2_dot + 10.0*h_dock2
    h_expr[2] = 2.0#-(u*cos(psi) - v*sin(psi)) + oy2*(-xn + ox1)


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
    model.u_labels = ['$n_1_d$ [N/s]', '$n_2_d$ [N/s]']
    model.t_label = '$t$ [s]'

    return model


def setup_trajectory_tracking(x0, N_horizon, Tf):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_heron_model()
    ocp.model = model

    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # Q_mat = 1*np.diag([0, 0, 0, 0, 0, 0, 1e-1, 1e-5])
    Q_mat = 1*np.diag([0, 1e3, 1e1, 1e0, 1e0, 1e-2, 1e-1, 1e-3])
    Q_mat_terminal = np.diag([1e3, 1e3, 1e2, 1e-1, 1e-1, 1e-1, 1e-1, 1e-3])
    R_mat = 1*np.diag([1e0, 1e-3])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat_terminal

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = x0


    ocp.parameter_values = np.array([0.0, 0.0, 0.0, 
                                     0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0])

    num_obs = 3
    ocp.constraints.uh = 1e10 * np.ones(num_obs)
    ocp.constraints.lh = np.zeros(num_obs)
    # h_expr = SX.zeros(num_obs,1)
    # h_expr[0] = (model.x[0]-ox1) ** 2 + (model.x[1] - oy1) ** 2 - or1**2
    # h_expr[1] = (model.x[0]-ox2) ** 2 + (model.x[1] - oy2) ** 2 - or2**2
    # ocp.model.con_h_expr = h_expr

    ocp.constraints.idxsh = np.array([0,1,2])
    ocp.constraints.idxsh_e = np.array([0,1,2])
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
    ocp.constraints.lbu = np.array([-100.0,-2.0])
    ocp.constraints.ubu = np.array([+100.0,+2.0])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-250.0, -9])
    ocp.constraints.ubx = np.array([250.0, 16])
    ocp.constraints.idxbx = np.array([6, 7])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 500
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
    # create an integrator with the same settings as used in the OCP solver.
    # acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver


