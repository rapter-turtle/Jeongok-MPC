from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt, log, exp, tanh

def export_heron_model_nlp2() -> AcadosModel:
    model_name = 'heron_nlp2'
    # constants
 
    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]
 
    # Mid speed
    # Xu_dot = 15.26
    # # Slow speed
    # Xu = 2.6
    # Xuu = 0.75
    # tau = 1.0
    # bu=  2.1/500.0

    Xu_dot = 30.26
    # Slow speed
    Xu = 2.6
    Xuu = 0.75
    tau = 1.0

    bu=  2.1/500.0

    # Yv = 0.042
    # Yvv = 0.0
    # Yr = 0.0
    # Nr = 0.1875
    # Nrr = 2.647
    # Nv = 0.0
    # b2 = 0.01/500.0
    # b3 = 1.4  

    Yv = 0.004
    Yr = 0.74
    Nr = 0.054
    Nv = 0.033
    Nrr = 1.4
    Yvv = 0.067
    b3 = 1.47
    b2 = 0.0089/500.0   

    F_bow = 0.007
    F_l = 4.0

    # set up states & controls
    xn   = SX.sym('xn')
    yn   = SX.sym('yn')
    psi  = SX.sym('psi')
    u    = SX.sym('u')
    v    = SX.sym('v')
    r    = SX.sym('r')

    delta  = SX.sym('delta')


    states = vertcat(xn, yn, psi, u, v, r, delta)

    delta_d  = SX.sym('delta_d')
    inputs  = vertcat(delta_d)

    # xdot
    xn_dot  = SX.sym('xn_dot')
    yn_dot  = SX.sym('yn_dot')
    psi_dot = SX.sym('psi_dot')
    u_dot   = SX.sym('u_dot')
    v_dot   = SX.sym('v_dot')
    r_dot   = SX.sym('r_dot')
    delta_dot   = SX.sym('delta_dot')

    # set up parameters
    thrust_button = SX.sym('thrust_button') 
    bow_button = SX.sym('bow_button') 
    or1 = SX.sym('or1') 
    ox1 = SX.sym('ox1') 
    oy1 = SX.sym('oy1') 


    p = vertcat(ox1, oy1, or1, thrust_button, bow_button)
    
    
    states_dot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, delta_dot)



    eps = 0.00001
    # dynamics
   
    thrust = 2.2*2.2
    f_expl = vertcat(u*cos(psi) - v*sin(psi),
                     u*sin(psi) + v*cos(psi),
                     r,
                     ( - Xu*u - Xuu * sqrt(u * u + eps) * u + thrust*thrust_button*cos(bu*delta))/(M + Xu_dot),
                     ( -Yv*v - Yvv * sqrt(v * v + eps) * v  - Yr*r + thrust*thrust_button*sin(b2*delta) + F_bow*bow_button),
                     ( - Nr*r - Nrr * sqrt(r * r + eps)*r - Nv*v - b3*thrust*thrust_button*sin(b2*delta) + F_l*F_bow*bow_button),
                     delta_d
                     )



    f_impl = states_dot - f_expl


    num_obs = 3
    # Dock position

    h_expr = SX.zeros(num_obs,1)
    h_expr[0] = -xn + 11.0
    h_expr[1] = -yn + 11.0
    h_expr[2] = -yn - 6*sin(psi)+ 11.0

    
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


def setup_trajectory_tracking_nlp2(x0, N_horizon, Tf, Q_mat, Q_mat_terminal, R_mat):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = export_heron_model_nlp2()
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


    ocp.parameter_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

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
    ocp.constraints.lbu = np.array([-50.0])
    ocp.constraints.ubu = np.array([+50.0])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-250.0])
    ocp.constraints.ubx = np.array([250.0])
    ocp.constraints.idxbx = np.array([6])

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


