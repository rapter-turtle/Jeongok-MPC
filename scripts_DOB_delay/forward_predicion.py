import numpy as np

def forward_prediction(state, control_input, delay, dt_control, disturbance, dt_sim=0.01):
    """
    state: np.array, 현재 상태 [x, y, psi, u, v, r, delta, F_eff]
    control_input: list of (delta_cmd, F_cmd) from past to current
    delay: float, 시간 지연 [s]
    dt_control: float, control update interval (예: 0.5s)
    disturbance: [du, dv, dr] 외란
    dt_sim: float, simulation step (default=0.01s)
    """

    # --- 파라미터 ---
    M, I = 1.0, 1.0
    Xu_dot, Xu, Xuu, bu = 22.43, 1.671, 0.481, 2.03/500.0
    Yv, Yr, Nr, Nrr, b2, b3 = 0.042, 0.0, 0.1875, 2.647, 0.01/500.0, 1.4
    eps = 1e-6

    # --- dynamics 함수 ---
    def dynamics(x, delta_cmd, F_cmd):
        x_pos, y_pos, psi, u, v, r, _, _ = x

        # Deadzone + thrust 비선형
        s, k = 25, 8
        a1 = a2 = 2.2**2
        b11 = b22 = 1.0
        T = ((1/(1+np.exp(s*F_cmd)))*(b11*F_cmd + np.tanh(k*F_cmd)*a1) +
             (1/(1+np.exp(-s*F_cmd)))*(b22*F_cmd + np.tanh(k*F_cmd)*a2))

        # Dynamics
        u_dot = (-Xu*u - Xuu*np.sqrt(u*u+eps)*u + T*np.cos(bu*delta_cmd)) / (M+Xu_dot) - disturbance[0]
        v_dot = (-Yv*v - Yr*r + T*np.sin(b2*delta_cmd)) / M - disturbance[1]
        r_dot = (-Nr*r - Nrr*np.sqrt(r*r+eps)*r - b3*T*np.sin(b2*delta_cmd)) / I - disturbance[2]

        # Kinematics
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


# import numpy as np

# def forward_prediction(state, control_input, delay, dt, disturbance):
#     """
#     state: np.array, 현재 상태 [x, y, psi, u, v, r, delta, F_eff]
#     control_input: list of (delta_cmd, F_cmd) from past to current
#     delay: float, 시간 지연 [s]
#     dt: float, 시뮬레이션 step
#     disturbance: [du, dv, dr] 외란
#     """

#     # --- 파라미터 ---
#     M = 1.0
#     I = 1.0

#     Xu_dot = 22.43
#     Xu = 1.671
#     Xuu = 0.481
#     bu = 2.03/500.0

#     Yv = 0.042
#     Yr = 0.0
#     Nr = 0.1875
#     Nrr = 2.647
#     b2 = 0.01/500.0
#     b3 = 1.4

#     eps = 1e-6


#     # --- 지연 시점부터 현재까지의 입력 sequence ---
#     input_seq = control_input

#     # --- dynamics 함수 정의 ---
#     def dynamics(x, delta_cmd, F_cmd):
#         x_pos, y_pos, psi, u, v, r, _, _ = x

#         # Deadzone + thrust 비선형
#         s, k = 25, 8
#         a1 = a2 = 2.2**2
#         b11 = 1.0
#         b22 = 1.0
#         T = ((1/(1+np.exp(s*F_cmd)))*(b11*F_cmd + np.tanh(k*F_cmd)*a1) +
#              (1/(1+np.exp(-s*F_cmd)))*(b22*F_cmd + np.tanh(k*F_cmd)*a2))

#         # Dynamics
#         u_dot = (-Xu*u - Xuu*np.sqrt(u*u+eps)*u + T*np.cos(bu*delta_cmd)) / (M+Xu_dot) - disturbance[0]
#         v_dot = (-Yv*v - Yr*r + T*np.sin(b2*delta_cmd)) / M - disturbance[1]
#         r_dot = (-Nr*r - Nrr*np.sqrt(r*r+eps)*r - b3*T*np.sin(b2*delta_cmd)) / I - disturbance[2]

#         # Kinematics
#         x_dot   = u*np.cos(psi) - v*np.sin(psi)
#         y_dot   = u*np.sin(psi) + v*np.cos(psi)
#         psi_dot = r

#         dx = np.zeros_like(x)
#         dx[0] = x_dot
#         dx[1] = y_dot
#         dx[2] = psi_dot
#         dx[3] = u_dot
#         dx[4] = v_dot
#         dx[5] = r_dot
#         return dx

#     # --- 초기 상태 설정 ---
#     x = state.copy()

#     # --- 입력 sequence를 RK4로 순서대로 적용 ---
#     for (delta_cmd, F_cmd) in input_seq:
#         k1 = dynamics(x, delta_cmd, F_cmd)
#         k2 = dynamics(x + 0.5*dt*k1, delta_cmd, F_cmd)
#         k3 = dynamics(x + 0.5*dt*k2, delta_cmd, F_cmd)
#         k4 = dynamics(x + dt*k3,     delta_cmd, F_cmd)

#         x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
#         x = x_next

#     # delay 반영된 입력 저장
#     x[6] = input_seq[-1][0]  # delay 구간의 첫 입력
#     x[7] = input_seq[-1][1]

#     return x
