from acados_setting_nlp1 import *
from acados_setting_nlp2 import *
from bow_CIA import *
from gurobipy import Model, GRB, QuadExpr
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from aura_msg.msg import MPCTraj, MPCState, ObsState
import math
import time
import numpy as np
from DOB import*

ship_state_x = (289577.66 + 291591.05)*0.5  # UTM X (easting)
ship_state_y = (4117065.30 + 4118523.52)*0.5  

traj_xy = (ship_state_x, ship_state_y)
offset = np.array([ship_state_x, ship_state_y])



class AuraMPC(Node):
    def __init__(self):
        super().__init__('aura_mpc')      
        # ROS setting
        self.publisher_ = self.create_publisher(Float64MultiArray, '/actuator_outputs', 10)
        self.ekf_sub = self.create_subscription(Float64MultiArray, '/ekf/estimated_state', self.ekf_callback, 10)
        self.mpcvis_pub = self.create_publisher(MPCTraj, '/mpc_vis', 10)
        self.DOB_pub = self.create_publisher(Float64MultiArray, '/DOB', 10)


        # Initial states and inputs
        self.x = ship_state_x
        self.y = ship_state_y
        self.p = self.u = self.v = self.r = 0.0
        self.delta = self.F = 0.0
        self.delta_pwm = self.F_pwm = 1500.0
        self.states = np.zeros(8)
        self.thr = 0.0
        self.del_thr_max = 0.5
        self.dob_thrust = 0.0
        
        # MPC parameter settings
        self.Tf = 15 # prediction time 4 sec
        self.N = 30 # prediction horizon
        self.con_dt = 0.5 # control sampling time

        Q_mat = 1*np.diag([1e0, 1e0, 1e3, 1e2, 1e2, 1e-2, 0, 0])
        Q_mat_terminal = 20*np.diag([1e1, 1e1, 1e3, 1e3, 1e3, 1e3, 0, 0])
        R_mat1 = 1*np.diag([1e-1, 1e-1, 1e1])
        R_mat2 = 1*np.diag([1e-1, 1e-1])        
        self.ocp_solver_nlp1 = setup_trajectory_tracking_nlp1(self.states, self.N, self.Tf, Q_mat, Q_mat_terminal, R_mat1)
        self.ocp_solver_nlp2 = setup_trajectory_tracking_nlp2(self.states, self.N, self.Tf, Q_mat, Q_mat_terminal, R_mat2)

        # DOB
        # DOB(state, state_estim, param_filtered, param_estim, dt):
        self.state_estim = np.array([0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0])
        self.param_estim = np.array([0.0, 0.0, 0.0])
        self.DOB_dt = 0.1

        self.a_dot_state = 0.0

        #CIA
        self.count_on_left = 0
        self.count_on_right = 0
        self.count_on_zero = 0
        self.bow_before = 0
        self.bow_switch = 0.0

        # reference trajectory generation
        self.A = 3.0
        self.B = 100.0
        self.C = 1.0
        
        self.k = 0
        self.create_timer(self.con_dt, self.run)
        self.create_timer(self.DOB_dt, self.run_DOB)
    
    def clamp(self, value, min_value, max_value):
        """Clamp the value to the range [min_value, max_value]"""
        return max(min_value, min(value, max_value))

    def convert_steering_to_pwm(self,steer):
        """Map steering value to PWM based on the given formula"""

        if steer >= 300.0:
            # Steer above 300 maps directly to PWM 2000
            return 2000.0
        elif 0 <= steer < 300.0:
            # Steer in the range [0, 300] maps linearly between PWM = 1500 and PWM = 2000
            return 1550.0 + (steer * 1.6667)
        elif -300.0 <= steer < 0:
            # Steer in the range [-300, 0] maps linearly between PWM = 1000 and PWM = 1500
            return 1450.0 + (steer * 1.6667)
        elif steer < -300.0:
            # Steer below -300 maps directly to PWM 1000
            return 1000.0

    # def convert_thrust_to_pwm(self, rpm_thrust, thr):
    #     """Convert thrust level to PWM signal"""
    #     # thr_new = np.sign(rpm_thrust - thr)*100*0.5 + thr

    #     # if (rpm_thrust-thr)*(rpm_thrust-thr_new)<0:
    #     #     thr_new = rpm_thrust
    #     thr_new = rpm_thrust
        

    #     if rpm_thrust <= 0.0 :
    #         thrust = -np.sqrt(-(rpm_thrust))
    #     elif rpm_thrust >= 0.0:
    #         thrust = np.sqrt(rpm_thrust)
    #     else:
    #         thrust = 0.0

    #     dob_thrust = thrust

    #     if thrust < 0.0:
    #         pwm = 3.9 * thrust + 1450.0
    #         return self.clamp(pwm, 1000.0, 1450.0), thr, dob_thrust  # Any value <= 0 thrust maps to PWM 1000
    #     else:
    #         # Calculate PWM based on the thrust
    #         pwm = 3.9 * thrust + 1550.0
    #         # You can switch the formula if needed, using the commented one
    #         # pwm = 5.0 * thrust + 1500
    #         return self.clamp(pwm, 1550.0, 2000.0), thr, dob_thrust  # Ensure PWM is within the bounds
    

    def convert_thrust_to_pwm(self, rpm_thrust, thr):
        """Convert thrust level to PWM signal"""
        # thr_new = np.sign(rpm_thrust - thr)*100*0.5 + thr

        # if (rpm_thrust-thr)*(rpm_thrust-thr_new)<0:
        #     thr_new = rpm_thrust
        thr_new = rpm_thrust
        # print(rpm_thrust)
        # deadzone = 22.0*22.0
        deadzone = 22.0*22.0
        threshold = 100.0
        # deadzone = 10.0*10.0
        if thr_new < threshold and thr_new > -threshold:
            thrust_2 = 0.0 
        else:        
            thrust_2 = deadzone*np.sign(thr_new) + thr_new
        

        if thrust_2 <= 0.0 :
            thrust = -np.sqrt(-(thrust_2))
        elif thrust_2 >= 0.0:
            thrust = np.sqrt(thrust_2)
        else:
            thrust = 0.0

        # print("thrust : ",thrust)
        
        dob_thrust = thrust

        if thrust < 0.0:
            pwm = 3.9 * thrust + 1450.0
            return self.clamp(pwm, 1000.0, 1450.0), thr, dob_thrust  # Any value <= 0 thrust maps to PWM 1000
        else:
            # Calculate PWM based on the thrust
            pwm = 3.9 * thrust + 1550.0
            # You can switch the formula if needed, using the commented one
            # pwm = 5.0 * thrust + 1500
            return self.clamp(pwm, 1550.0, 2000.0), thr, dob_thrust  # Ensure PWM is within the bounds
    
    def ekf_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
        self.states = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.delta, self.F])
        self.a_dot_state = msg.data[6]

    def yaw_discontinuity(self, ref):
        """Handle yaw angle discontinuities."""
        flag = [0.0] * 3
        flag[0] = abs(self.states[2] - ref)
        flag[1] = abs(self.states[2] - (ref - 2 * math.pi))
        flag[2] = abs(self.states[2] - (ref + 2 * math.pi))
        min_element_index = flag.index(min(flag))

        if min_element_index == 0:
            ref = ref
        elif min_element_index == 1:
            ref = ref - 2 * math.pi
        elif min_element_index == 2:
            ref = ref + 2 * math.pi
        return ref

 
    def run(self):
        k = self.k # -> 현재 시간을 index로 표시 -> 그래야 ref trajectory설정가능(******** todo ********)
                
        t = time.time()             
        dock_x = 0.0
        dock_y = 0.0
        dock_psi = 0.0
        #################################### 1st NLP ####################################
        
        ##### Obstacle Position ######
        obs_pos = np.array([self.A, self.B, self.C,  # Obstacle-1: x, y, radius
                            self.A, self.B, self.C]) # Obstacle-2: x, y, radius
                            # self.param_filtered[0], self.param_filtered[1], self.param_filtered[2]]) # Obstacle-2: x, y, radius
        
        for j in range(self.N+1):
            self.ocp_solver_nlp1.set(j, "p", obs_pos)
    
        
        # preparation phase
        self.ocp_solver_nlp1.options_set('rti_phase', 1)
        status = self.ocp_solver_nlp1.solve()
        t_preparation = self.ocp_solver_nlp1.get_stats('time_tot')

        # set initial state
        self.ocp_solver_nlp1.set(0, "lbx", self.states)
        self.ocp_solver_nlp1.set(0, "ubx", self.states)

        # feedback phase
        self.ocp_solver_nlp1.options_set('rti_phase', 2)
        status = self.ocp_solver_nlp1.solve()
        t_feedback = self.ocp_solver_nlp1.get_stats('time_tot')

        # obtain mpc input
        bow_array = np.zeros(self.N)
        for j in range(self.N):
            con = self.ocp_solver_nlp1.get(j,"u")
            # print(con)
            bow_array[j] = con[2]
        # bow_array[0] = self.bow_switch    
            # print(self.ocp_solver_nlp1.get(j,"u"))
        # print(bow_array)

        # self.get_logger().info(f"MPC Computation Time: {t_preparation + t_feedback:.4f}s")


        #################################### CIA ####################################
        
        model = Model("bow_mapping")
        dwell_time = 2.0
        stop_dwell_time = 2.0
        CIA_bow_array = bow_mapping(model, bow_array, self.bow_switch, self.con_dt, len(bow_array), dwell_time, stop_dwell_time)
        
       

        dwell_len = int(dwell_time / self.con_dt)
        stop_dwell_len = int(stop_dwell_time / self.con_dt)
        
        if self.count_on_left <= dwell_len - 1 and self.bow_switch > 0.1:
            self.count_on_left += 1
            start_idx = dwell_len - self.count_on_left 
            input_bow_array = bow_mapping(model, bow_array[start_idx:], self.bow_switch,self.con_dt, len(bow_array[start_idx:]),
                                        dwell_time, stop_dwell_time)
            CIA_bow_array = np.concatenate([np.ones(start_idx), input_bow_array])

            self.count_on_right = 0
            self.count_on_zero = 0


        elif self.count_on_right <= dwell_len - 1 and self.bow_switch < -0.1:
            self.count_on_right += 1
            start_idx = dwell_len - self.count_on_right 
            input_bow_array = bow_mapping(model, bow_array[start_idx:], self.bow_switch,self.con_dt, len(bow_array[start_idx:]),
                                        dwell_time, stop_dwell_time)
            CIA_bow_array = np.concatenate([-np.ones(start_idx), input_bow_array])

            self.count_on_left = 0
            self.count_on_zero = 0

        elif self.count_on_zero <= stop_dwell_len - 1 and self.bow_switch == 0.0:
            self.count_on_zero += 1
            start_idx = dwell_len - self.count_on_zero 
            input_bow_array = bow_mapping(model, bow_array[start_idx:], self.bow_switch,self.con_dt, len(bow_array[start_idx:]),
                                        dwell_time, stop_dwell_time)
            CIA_bow_array = np.concatenate([np.zeros(start_idx), input_bow_array])

            self.count_on_left = 0
            self.count_on_right = 0


        elif self.count_on_right >= dwell_len and self.bow_switch > 0.1:
            self.count_on_right = 0

        elif self.count_on_left >= dwell_len and self.bow_switch < -0.1:
            self.count_on_left = 0

        elif self.count_on_left >= stop_dwell_len and self.bow_switch != -0.1:
            self.count_on_zero = 0

        # print(bow_array)          
        print(CIA_bow_array) 


       
        #################################### 2nd NLP ####################################
        
        ##### Obstacle Position ######
        for j in range(self.N):

            obs_pos = np.array([CIA_bow_array[j], 0.0]) 
            self.ocp_solver_nlp2.set(j, "p", obs_pos)
    
        # preparation phase
        self.ocp_solver_nlp2.options_set('rti_phase', 1)
        status = self.ocp_solver_nlp2.solve()
        t_preparation = self.ocp_solver_nlp2.get_stats('time_tot')

        # set initial state
        self.ocp_solver_nlp2.set(0, "lbx", self.states)
        self.ocp_solver_nlp2.set(0, "ubx", self.states)

        # feedback phase
        self.ocp_solver_nlp2.options_set('rti_phase', 2)
        status = self.ocp_solver_nlp2.solve()
        t_feedback = self.ocp_solver_nlp2.get_stats('time_tot')

        # obtain mpc input
        del_con = self.ocp_solver_nlp2.get(0, "u")
        self.delta += del_con[0]*self.con_dt
        self.F += del_con[1]*self.con_dt
        self.bow_switch = CIA_bow_array[0]

        # self.get_logger().info(f"MPC Computation Time: {t_preparation + t_feedback:.4f}s")


        self.delta_pwm = self.convert_steering_to_pwm(self.delta)
        self.F_pwm, self.thr, self.dob_thrust = self.convert_thrust_to_pwm(self.F*100.0, self.thr)                        
        actuator_msg = Float64MultiArray()
        actuator_msg.data = [self.delta_pwm, self.F_pwm, self.bow_switch, 0.0]
        # print("thrust d : ",del_con[1], "F : ", self.F*100.0)


        self.publisher_.publish(actuator_msg)                                
        
        print("Time : ",time.time()-t)

        # Publish predicted states and reference states
        mpc_data_stack = MPCTraj()
        # mpc_data_stack.header.stamp = self.get_clock()
        mpc_data_stack.pred_num = float(self.N)
        mpc_data_stack.sampling_time = self.con_dt
        mpc_data_stack.cpu_time = t_preparation + t_feedback	
        mpc_data_stack.ref_num = 0.0	
        mpc_data_stack.ref_dt = 200.0	
        mpc_data_stack.traj_x = dock_x + ship_state_x
        mpc_data_stack.traj_y = dock_y + ship_state_y
        mpc_data_stack.theta = dock_psi	
        mpc_data_stack.a = self.A
        mpc_data_stack.b = self.B	
        mpc_data_stack.c = self.C	
        
        for j in range(self.N+1):
            mpc_pred = MPCState()
            mpc_ref = MPCState()
            mpc_pred.x = self.ocp_solver_nlp2.get(j, "x")[0]+offset[0]
            mpc_pred.y = self.ocp_solver_nlp2.get(j, "x")[1]+offset[1]
            mpc_pred.p = self.ocp_solver_nlp2.get(j, "x")[2]
            mpc_pred.u = self.ocp_solver_nlp2.get(j, "x")[3]
            mpc_pred.v = self.ocp_solver_nlp2.get(j, "x")[4]
            mpc_pred.r = self.ocp_solver_nlp2.get(j, "x")[5]
            mpc_pred.delta = self.ocp_solver_nlp2.get(j, "x")[6]
            mpc_pred.f = self.ocp_solver_nlp2.get(j, "x")[7]
            mpc_data_stack.state.append(mpc_pred)            
            # print(mpc_pred.u)
            mpc_ref.x = self.ocp_solver_nlp2.get(j, "x")[0]+offset[0]
            mpc_ref.y = self.ocp_solver_nlp2.get(j, "x")[1]+offset[1]
            mpc_ref.p = 0.0
            mpc_ref.u = 0.0
            mpc_ref.v = 0.0
            mpc_ref.r = 0.0
            mpc_ref.delta = 0.0
            mpc_ref.f = 0.0
            mpc_data_stack.ref.append(mpc_ref)            


        obs_state = ObsState()
        obs_state.x   = 0.0+offset[0]
        obs_state.y   = 0.0+offset[1]
        obs_state.rad = 0.0
        mpc_data_stack.obs.append(obs_state)
        obs_state = ObsState()
        obs_state.x   = 0.0
        obs_state.y   = 0.0
        obs_state.rad = 0.0
        mpc_data_stack.obs.append(obs_state)        

        self.mpcvis_pub.publish(mpc_data_stack)
        


    def run_DOB(self):
        dob_state = self.states
        dob_state[7] = self.dob_thrust
        self.state_estim, self.param_estim, self.param_filtered = DOB(self.states, self.state_estim, self.param_filtered, self.param_estim, self.DOB_dt)
             
        DOB_msg = Float64MultiArray()
        DOB_msg.data = [self.param_filtered[0], self.param_filtered[1], self.param_filtered[2]]
        self.DOB_pub.publish(DOB_msg)                     
        


def main(args=None):
    rclpy.init(args=args)
    node = AuraMPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()