import numpy as np

def forward_prediction(state, control_input, delay, dt_control, disturbance, rff_params, dt_sim=0.01):
    """
    state: np.array, 현재 상태 [x, y, psi, u, v, r, delta, F_eff]
    control_input: list of (delta_cmd, F_cmd) from past to current
    delay: float, 시간 지연 [s]
    dt_control: float, control update interval (예: 0.5s)
    disturbance: [du, dv, dr] 외란 (현재는 사용 안 함)
    dt_sim: float, simulation step (default=0.01s)
    """

    # --- 파라미터 ---
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

    eps = 1e-6

    def dynamics(x, delta_cmd, F_cmd):
        x_pos, y_pos, psi, u, v, r, _, _ = x

        # thrust nonlinear model
        s, k = 25, 8
        a1 = a2 = 2.2**2
        b11 = b22 = 1.0
        T = ((1/(1+np.exp(s*F_cmd)))*(b11*F_cmd + np.tanh(k*F_cmd)*a1) +
             (1/(1+np.exp(-s*F_cmd)))*(b22*F_cmd + np.tanh(k*F_cmd)*a2))

        # -------------------------
        #  RFF DISTURBANCE MODEL
        # -------------------------
        # surge
        z_u = np.array([u, F_cmd, delta_cmd])
        phi_u = np.cos(rff_params["w_u"] @ z_u + rff_params["b_u"])
        h_u = (rff_params["alpha_u"] @ phi_u) / len(rff_params["alpha_u"])

        # sway/yaw
        z_lat = np.array([v, r, F_cmd, delta_cmd])
        phi_lat = np.cos(rff_params["w_lat"] @ z_lat + rff_params["b_lat"])
        h_v = (rff_params["alpha_v"] @ phi_lat) / len(rff_params["alpha_v"])
        h_r = (rff_params["alpha_r"] @ phi_lat) / len(rff_params["alpha_r"])

        # -------------------------
        # nominal dynamics + disturbance
        # -------------------------
        u_dot = (-Xu*u - Xuu*np.sqrt(u*u+eps)*u + alpha1*T*np.cos(alpha2*delta_cmd)) + h_u
        v_dot = (-Yv*v - Yr*r - Yvv*np.sqrt(v*v+eps)*v + alpha3*T*np.sin(alpha2*delta_cmd)) + h_v
        r_dot = (-Nr*r - Nv*v - Nrr*np.sqrt(r*r+eps)*r - alpha4*T*np.sin(alpha2*delta_cmd)) + h_r

        # kinematics
        x_dot   = u*np.cos(psi) - v*np.sin(psi)
        y_dot   = u*np.sin(psi) + v*np.cos(psi)
        psi_dot = r

        dx = np.zeros_like(x)
        dx[0], dx[1], dx[2] = x_dot, y_dot, psi_dot
        dx[3], dx[4], dx[5] = u_dot, v_dot, r_dot
        return dx

    # --- 초기 상태 ---
    x = state.copy()

    # --- 입력 sequence 순서대로 적용 ---
    steps_per_control = int(dt_control / dt_sim)
    for (delta_cmd, F_cmd) in control_input:
        for _ in range(steps_per_control):
            k1 = dynamics(x, delta_cmd, F_cmd)
            k2 = dynamics(x + 0.5*dt_sim*k1, delta_cmd, F_cmd)
            k3 = dynamics(x + 0.5*dt_sim*k2, delta_cmd, F_cmd)
            k4 = dynamics(x + dt_sim*k3,     delta_cmd, F_cmd)
            x = x + (dt_sim/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # delay 반영된 입력 저장 (여기서는 가장 마지막 입력)
    x[6] = control_input[-1][0]
    x[7] = control_input[-1][1]

    return x
