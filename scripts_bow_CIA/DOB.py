import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time
import numpy as np

   
def DOB(state, state_estim, param_filtered, param_estim, dt):


    M = 1.0  # Mass [kg]
    I = 1.0   # Inertial tensor [kg m^2]

    Xu_dot = 30.26
    Xu = 2.6
    Xuu = 0.75
    tau = 1.0
    bu=  2.1/500.0


    Yv = 0.004
    Yr = 0.74
    Nr = 0.054
    Nv = 0.033
    Nrr = 1.4
    Yvv = 0.067
    b3 = 1.47
    b2 = 0.0089/500.0    

    w_cutoff = 0.5
    gain = -1.0


    # set up states & controls
    u    = state[3]
    v    = state[4]
    r    = state[5]
    delta  = state[6]
    F  = state[7]
    eps = 0.00001

    a = 1.0
    # Deadzone
    # F = F_cmd 
    if F >= 0.0:
        a = 1.0
    else:
        a = -1.0


    f_usv = np.array([(- Xu*u - Xuu * np.sqrt(u * u + eps) * u + 0.01*a*F*F*np.cos(bu*delta))/(M + Xu_dot),
                    ( -Yv*v - Yvv * np.sqrt(v * v + eps) * v - Yr*r + 0.01*a*F*F*np.sin(b2*delta)),
                    ( - Nr*r - Nrr * np.sqrt(r * r + eps) * r - Nv*v - b3*0.01*a*F*F*np.sin(b2*delta))
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


