import numpy as np
import time

def f(state, F, B, delta):
    
    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]
 
    # Mid speed
    Xu_dot = 30.26
    Xu = 2.6
    Xuu = 0.75
    tau = 1.0
    bu=  2.1/500.0

    F_bow = 0.007
    F_l = 4.0

    Yv = 0.004
    Yr = 0.74
    Nr = 0.054
    Nv = 0.033
    Nrr = 1.4
    Yvv = 0.067
    b3 = 1.47
    b2 = 0.0089/500.0    

    psi = state[2]
    u = state[3]
    v = state[4]
    r = state[5]

    df = np.array([u*np.cos(psi) - v*np.sin(psi),
                     u*np.sin(psi) + v*np.cos(psi),
                     r,
                     ( - Xu*u - Xuu * np.sqrt(u * u) * u + 2.2*2.2*F*np.cos(bu*delta))/(M + Xu_dot),
                     ( -Yv*v - Yvv * np.sqrt(v * v) * v - Yr*r + 2.2*2.2*F*np.sin(b2*delta) + F_bow*B),
                     ( - Nr*r - Nrr * np.sqrt(r * r)*r - Nv*v - b3*2.2*2.2*F*np.sin(b2*delta) + F_l*F_bow*B),
                     ])
    return df



def obj_calc(gear_array, bow_array, state_history, N):
    """
    Weighted state-tracking objective over first 6 states.
    Weights are fixed inside this function.
    """
    dt = 0.5

    # Define fixed weights for [x, y, psi, u, v, r]
    # 👉 adjust these values as you prefer
    state_weights = np.array([5.0, 5.0, 2.0, 0.0, 0.0, 0.0])

    objective_function = 0.0

    next_state = np.array(state_history[0], dtype=float).copy()

    for j in range(N):
        # 1) weighted squared error (only 0..5 indices)
        ref_state6 = np.asarray(state_history[j][:6], dtype=float)
        err6 = (next_state[:6] - ref_state6)
        objective_function += float(np.dot(state_weights, err6 * err6))

        # 2) propagate dynamics
        delta = state_history[j][6]
        df = f(next_state, float(gear_array[j]), float(bow_array[j]), delta)

        # update only first 6 states
        next_state[:6] = next_state[:6] + dt * df[:6]

    return float(objective_function)
