import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time
import numpy as np


def DOB(state, state_estim, param_filtered, param_estim, dt):

 
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

    w_cutoff = 0.5
    gain = -1.0


    # set up states & controls
    u    = state[3]
    v    = state[4]
    r    = state[5]
    delta  = state[6]
    F  = state[7]
    eps = 0.00001

    # Deadzone
    # F = F_cmd 
    a = 0
    if F < 0:
        a = -1
    else:
        a = 1


    f_usv = np.array([(- Xu*u - Xuu * np.sqrt(u * u + eps) * u + a*0.01*F*F*np.cos(bu*delta))/(M + Xu_dot),
                    ( -Yv*v - Yvv * np.sqrt(v * v + eps) * v - Yr*r + a*0.01*F*F*np.sin(b2*delta)),
                    ( - Nr*r - Nrr * np.sqrt(r * r + eps) * r - a*b3*0.01*F*F*np.sin(b2*delta))
                    ])



    uvr = np.array([u, v, r])

    x_error = state_estim - uvr 
    
    
    xdot = param_estim + f_usv + gain*x_error

    x_t_plus = xdot*dt + state_estim
    


    pi = (1/gain)*(np.exp(gain*dt)-1.0)
    param_estim = -np.exp(gain*dt)*x_error/pi
    
    before_param_filtered = param_filtered
    param_filtered = before_param_filtered*math.exp(-w_cutoff*dt) - param_estim*(1-math.exp(-w_cutoff*dt))


    
    return x_t_plus, param_estim, param_filtered


