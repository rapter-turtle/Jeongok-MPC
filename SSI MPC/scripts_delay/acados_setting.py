from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg
import numpy as np
from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, sqrt, exp, tanh

def export_heron_model() -> AcadosModel:
    model_name = 'heron'
    # constants

    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]

    Xu = 0.0845
    Xuu = 0.0195
    Yv = 0.0485
    Yvv = 0.0988
    Yr = 0.151
    Nr = 0.6939
    Nrr = 0.0
    Nv = 0.0
    alpha1 = 0.0452
    alpha2 = 2.0/500.0
    alpha3 = 0.0188
    alpha4 = 0.0193      

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
    states_dot = vertcat(xn_dot, yn_dot, psi_dot, u_dot, v_dot, r_dot, delta_dot, F_dot)

    
    M_feat = 50      # number of random features
    dim_z  = 5       # z = [u, v, r, delta, F]

    # α_u, α_v, α_r  (각 M_feat)
    alpha_u = SX.sym('alpha_u', M_feat)
    alpha_v = SX.sym('alpha_v', M_feat)
    alpha_r = SX.sym('alpha_r', M_feat)

    # w: (M_feat * dim_z,)  flatten한 상태
    w = SX.sym('w', M_feat * dim_z)

    # b: (M_feat,)
    b = SX.sym('b', M_feat)

    # 전체 parameter vector
    p = vertcat(alpha_u, alpha_v, alpha_r, w, b)

    s = 25
    k = 8
    a1 = 2.2*2.2
    a2 = 2.2*2.2
    b11 = 1.0
    b22 = 1.0 
 
    T = ((1/(1+exp(s*F)))*(b11*F + tanh(k*F)*a1) + (1/(1+exp(-s*F)))*(b22*F + tanh(k*F)*a2))


    eps = 0.00001


    # -------------------------
    # RFF feature φ(z) 계산
    # -------------------------
    # z = [u, v, r, delta, F]
    z = vertcat(u, v, r, delta, F)

    phi = SX.zeros(M_feat, 1)

    for i in range(M_feat):
        # w_i: dim_z 길이의 부분벡터
        dot_wz = 0
        for j in range(dim_z):
            dot_wz = dot_wz + w[i*dim_z + j] * z[j]
        phi[i] = cos(dot_wz + b[i])

    # h_u, h_v, h_r 계산
    h_u = (alpha_u.T @ phi) / M_feat
    h_v = (alpha_v.T @ phi) / M_feat
    h_r = (alpha_r.T @ phi) / M_feat



    f_expl = vertcat(u*cos(psi) - v*sin(psi),
                    u*sin(psi) + v*cos(psi),
                    r,
                    ( - Xu*u - Xuu * sqrt(u * u + eps) * u + alpha1*T*cos(alpha2*delta)) + h_u,
                    ( -Yv*v - Yr*r - Yvv * sqrt(v * v + eps) * v + alpha3*T*sin(alpha2*delta)) + h_v,
                    ( - Nr*r - Nv*v - Nrr * sqrt(r * r + eps) * r - alpha4*T*sin(alpha2*delta)) + h_r,
                    delta_d,
                    F_d
                    )


    f_impl = states_dot - f_expl


    num_obs = 2
    alpha = 0.5
 


    h_expr = SX.zeros(num_obs,1)
    h_expr[0] = 2.0#h1
    h_expr[1] = 2.0#h2
    # h_expr[0] = h1*alpha + h1_dot
    # h_expr[1] = h2*alpha + h2_dot
    


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

    Q_mat = 2*np.diag([1e3, 1e3, 1e-2, 0, 0.0, 0.0, 1e-1, 1e0])
    R_mat = 2*np.diag([1e2, 1e1])

    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
    ocp.cost.W_e = Q_mat

    ocp.model.cost_y_expr = vertcat(model.x, model.u)
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.yref  = np.zeros((ny, ))
    ocp.cost.yref_e = np.zeros((ny_e, ))

    ocp.constraints.x0 = x0

    M_feat = 50
    dim_z  = 5
    total_param_dim = 3*M_feat + M_feat*dim_z + M_feat  # 3M + M*dim_z + M
    ocp.parameter_values = np.zeros(total_param_dim)



    num_obs = 2
    ocp.constraints.uh = 1e10 * np.ones(num_obs)
    ocp.constraints.lh = np.zeros(num_obs)


    ocp.constraints.idxsh = np.array([0,1])
    ocp.constraints.idxsh_e = np.array([0,1])
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
    ocp.constraints.lbu = np.array([-100,-1.0])
    ocp.constraints.ubu = np.array([+100,+1.0])
    # ocp.constraints.lbu = np.array([-200,-1.0])
    # ocp.constraints.ubu = np.array([+200,+1.0])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([-250, 0.0])
    ocp.constraints.ubx = np.array([250, 13.5])
    # ocp.constraints.lbx = np.array([-250, 0.0])
    # ocp.constraints.ubx = np.array([250, 20])    
    ocp.constraints.idxbx = np.array([6, 7])

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.sim_method_newton_iter = 100
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf
    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)
    # create an integrator with the same settings as used in the OCP solver.
    # acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    return acados_ocp_solver


