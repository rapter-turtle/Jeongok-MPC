import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time
import numpy as np
from costum_for_numpy import GPSurgeModelNumpy


def DOB(con,state, state_estim, param_filtered, param_estim, dt):

 
    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]

    GP_NPZ_PATH = "/home/user/aura_ws/src/aura_mpc/GP_MPC/surge_model/surge_acados.npz"
    gp_surge_np = GPSurgeModelNumpy(GP_NPZ_PATH)

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

    w_cutoff = 1.0
    gain = -1.0


    # set up states & controls
    u    = state[3]
    v    = state[4]
    r    = state[5]
    delta  = con[0]
    # F_eff  = state[7]
    F_eff  = con[1]
    eps = 0.00001
 
    # # F = F_cmd 
    # Deadzone
    # s = 25
    # k = 8
    # a1 = 2.2*2.2
    # a2 = 2.2*2.2
    # b11 = 1.0
    # b22 = 1.0
    # T = ((1/(1+np.exp(s*F_eff)))*(b11*F_eff + np.tanh(k*F_eff)*a1) + (1/(1+np.exp(-s*F_eff)))*(b22*F_eff + np.tanh(k*F_eff)*a2))
    T = F_eff

    # f_usv = np.array([( - Xu*u - Xuu * np.sqrt(u * u + eps) * u + alpha1*T*np.cos(alpha2*delta)),
    f_usv = np.array([( gp_surge_np(u, T, delta)) ,
                    ( -Yv*v - Yr*r - Yvv * np.sqrt(v * v + eps) * v + alpha3*T*np.sin(alpha2*delta)) ,
                    ( - Nr*r - Nv*v - Nrr * np.sqrt(r * r + eps) * r - alpha4*T*np.sin(alpha2*delta))
                    ])


    uvr = np.array([u, v, r])

    x_error = state_estim - uvr 
    
    
    xdot = param_estim + f_usv + gain*x_error

    x_t_plus = xdot*dt + state_estim
    


    pi = (1/gain)*(np.exp(gain*dt)-1.0)
    param_estim = -np.exp(gain*dt)*x_error/pi
    
    before_param_filtered = param_filtered
    param_filtered = before_param_filtered*math.exp(-w_cutoff*dt) - param_estim*(1-math.exp(-w_cutoff*dt))

    # print(param_filtered)
    
    return x_t_plus, param_estim, param_filtered


