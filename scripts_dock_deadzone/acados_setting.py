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
    F  = SX.sym('F')

    states = vertcat(xn, yn, psi, u, v, r, delta, F)

    delta_d  = SX.sym('delta_d')
    F_d  = SX.sym('F_d')
    inputs  = vertcat(delta_d, F_d)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    delta_dot   = SX.sym('delta_dot')
    F_dot   = SX.sym('F_dot')

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
    
    
    states_dot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, delta_dot, F_dot)


    # Deadzone
    s = 25
    k = 8
    # s = 25
    # k = 1
    a1 = 2.2*2.2
    a2 = 2.2*2.2
    b11 = 1.0
    b22 = 1.0

    # F = F_cmd 
    T = 0.5*((1/(1+exp(s*F)))*(b11*F + tanh(k*F)*a1) + (1/(1+exp(-s*F)))*(b22*F + tanh(k*F)*a2))

    eps = 0.00001
    # dynamics
    f_expl = vertcat(u*cos(psi) - v*sin(psi),
                     u*sin(psi) + v*cos(psi),
                     r,
                     ( - Xu*u - Xuu * sqrt(u * u + eps) * u + T*cos(bu*delta))/(M + Xu_dot) - du,
                     ( -Yv*v - Yr*r + T*sin(b2*delta)) - dv,
                     ( - Nr*r - b3*T*sin(b2*delta)) - dr,
                     delta_d,
                     F_d
                     )

    # f_expl = vertcat(u*cos(psi) - v*sin(psi),
    #                  u*sin(psi) + v*cos(psi),
    #                  r,
    #                  ( - Xu*u - Xuu * sqrt(u * u + eps) * u + ((1/(1+exp(s*F)))*(b11*F + tanh(k*F)*a1) + (1/(1+exp(-s*F)))*(b22*F + tanh(k*F)*a2))*cos(bu*delta))/(M + Xu_dot) - du,
    #                  ( -Yv*v - Yr*r + ((1/(1+exp(s*F)))*(b11*F + tanh(k*F)*a1) + (1/(1+exp(-s*F)))*(b22*F + tanh(k*F)*a2))*sin(b2*delta)) - dv,
    #                  ( - Nr*r - b3*((1/(1+exp(s*F)))*(b11*F + tanh(k*F)*a1) + (1/(1+exp(-s*F)))*(b22*F + tanh(k*F)*a2))*sin(b2*delta)) - dr,
    #                  delta_d,
    #                  F_d
    #                  )
    

    f_impl = states_dot - f_expl


    num_obs = 3

    #docking
    head_l = 4.0
    xh_dot = u*cos(psi) - v*sin(psi) - r*head_l*sin(psi)
    yh_dot = u*sin(psi) + v*cos(psi) + r*head_l*cos(psi)
    xh = xn + head_l*cos(psi)
    yh = yn + head_l*sin(psi)
    
    # Rot
    x_rot = xh*cos(or1) - yh*sin(or1)
    y_rot = xh*sin(or1) + yh*cos(or1)
    x_rot_dot = xh_dot*cos(or1) - yh_dot*sin(or1)
    y_rot_dot = xh_dot*sin(or1) + yh_dot*cos(or1)

    # 1-tanh2
    # h_dock1 = 2.0 + 10.0*(1.0 - tanh(0.5*(xh - ox1 + 4))) - (yh - oy1)
    # h_dock2 = 2.0 + 10.0*(1.0 - tanh(0.5*(xh - ox1 + 4))) + (yh - oy1)
    # h_dock1_dot = -xh_dot*10.0*0.5*(1-tanh(0.5*(xh - ox1 + 4))*tanh(0.5*(xh - ox1 + 4))) + yh_dot
    # h_dock2_dot = -xh_dot*10.0*0.5*(1-tanh(0.5*(xh - ox1 + 4))*tanh(0.5*(xh - ox1 + 4))) - yh_dot
    
    dock_end_x = ox1*cos(or1) - oy1*sin(or1)
    dock_end_y = ox1*sin(or1) + oy1*cos(or1)

    h_dock1 = 2.0 + 10.0*(1.0 - tanh(0.5*(x_rot - dock_end_x + 4))) - (y_rot - dock_end_y)
    h_dock2 = 2.0 + 10.0*(1.0 - tanh(0.5*(x_rot - dock_end_x + 4))) + (y_rot - dock_end_y)
    h_dock1_dot = -x_rot_dot*10.0*0.5*(1-tanh(0.5*(x_rot - dock_end_x + 4))*tanh(0.5*(x_rot - dock_end_x + 4))) + y_rot_dot
    h_dock2_dot = -x_rot_dot*10.0*0.5*(1-tanh(0.5*(x_rot - dock_end_x + 4))*tanh(0.5*(x_rot - dock_end_x + 4))) - y_rot_dot
    
    
    h_expr = SX.zeros(num_obs,1)
    h_expr[0] = 10*h_dock1 + h_dock1_dot 
    h_expr[1] = 10*h_dock2 + h_dock2_dot
    h_expr[2] = 2.0#-x_rot + dock_end_x + 5



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

    # Q_mat = 1*np.diag([0, 0, 0, 0, 0, 0, 1e-2, 1e-3])
    Q_mat = 1*np.diag([1e0, 1e0, 1e0, 1e1, 1e1, 1e0, 0, 0])
    Q_mat_terminal = 40*np.diag([1e1, 1e1, 1e2, 1e3, 1e3, 1e3, 0, 0])
    R_mat = 1*np.diag([1e-1, 1e-1])

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


    ocp.constraints.idxsh = np.array([0,1,2])
    ocp.constraints.idxsh_e = np.array([0,1,2])
    Zh = 1e5 * np.ones(num_obs)
    zh = 1e5 * np.ones(num_obs)
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
    ocp.constraints.lbu = np.array([-100,-4.0])
    ocp.constraints.ubu = np.array([+100,+4.0])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-250, -9.0])
    ocp.constraints.ubx = np.array([250, 9.0])
    ocp.constraints.idxbx = np.array([6, 7])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
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

