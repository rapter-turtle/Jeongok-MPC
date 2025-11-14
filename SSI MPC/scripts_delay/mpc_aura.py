from gen_ref import *
from acados_setting import *
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from aura_msg.msg import MPCTraj, MPCState, ObsState
import math
import time
import numpy as np
from forward_predicion import*

ship_state_x = 290245.0#(289577.66 + 291591.05)*0.5  # UTM X (easting)
ship_state_y = 4118000.0#(4117065.30 + 4118523.52)*0.5  

traj_xy = (ship_state_x, ship_state_y)
offset = np.array([ship_state_x, ship_state_y])


class AuraMPC(Node):
    def __init__(self):
        super().__init__('aura_mpc')      
        # ROS setting
        self.publisher_ = self.create_publisher(Float64MultiArray, '/actuator_outputs', 10)
        self.ekf_sub = self.create_subscription(Float64MultiArray, '/ekf/estimated_state', self.ekf_callback, 10)
        self.mpcvis_pub = self.create_publisher(MPCTraj, '/mpc_vis', 10)
        # self.DOB_pub = self.create_publisher(Float64MultiArray, '/DOB_check', 10)
        self.DOB_pub = self.create_publisher(Float64MultiArray, '/DOB', 10)


        # Initial states and inputs
        self.x = ship_state_x
        self.y = ship_state_y
        self.p = self.u = self.v = self.r = 0.0
        self.delta = self.F = self.F_eff = 0.0
        self.delta_pwm = self.F_pwm = 1500.0
        self.states = np.zeros(8)
        self.thr = 0.0
        
        # MPC parameter settings
        self.Tf = 20 # prediction time 4 sec
        self.N = 40 # prediction horizon 
        self.con_dt = 0.5 # control sampling time
        self.ocp_solver = setup_trajectory_tracking(self.states, self.N, self.Tf)

        # Time delay
        self.delay = 3.0   # 2초 delay
        self.delay_step = int(self.delay / self.con_dt)
        self.input_history = np.zeros((self.delay_step, 2))
        self.now_delta = 0.0
        self.new_input = np.array([0.0,0.0])

        # DOB(state, state_estim, param_filtered, param_estim, dt):
        self.state_estim = np.array([0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0])
        self.param_estim = np.array([0.0, 0.0, 0.0])
        self.DOB_dt = 0.1
        self.dob_thrust = 0.0
        self.start_time = time.time()

        self.DOB_switch = False

        # RFF initialization (add into __init__)
        self.M = 50
        self.dim_z = 5  # define according to your z
        self.w, self.b = self.init_rff_parameters(self.M, self.dim_z, sigma_w=0.01)
        self.alpha_u = np.zeros(self.M)
        self.alpha_v = np.zeros(self.M)
        self.alpha_r = np.zeros(self.M)
        self.states_prev = np.zeros(8)  # previous state



        # reference trajectory generation
        # V1
        # self.A = 150.0
        # self.B = 100.0
        # self.C = 35.0
        # # V2
        # self.A = 70.0
        # self.B = 55.0
        # self.C = 18.0
        # V3
        self.A = 120.0
        self.B = 85.0
        self.C = 28.0
        self.theta = 0.0#np.pi
        self.plot_traj_xy = (ship_state_x-300.0, ship_state_y+0)
        
        self.ref_dt = 0.01
        self.ref_iter = int(self.con_dt/self.ref_dt)
        # self.reference = generate_figure_eight_trajectory_con(1000, self.ref_dt, self.plot_traj_xy, self.theta, self.A, self.B, self.C) # reference dt = 0.01 sec, 1000 sec trajectory generation
        
        # print("speed : ", self.reference[0,3])
        # self.reference = generate_figure_eight_trajectory_con(1000, self.ref_dt) # reference dt = 0.01 sec, 1000 sec trajectory generation
        # self.reference = self.reference[::self.ref_iter,:]
        self.ref_check = np.zeros(self.N+1)
        
        self.k = 0
        self.create_timer(self.con_dt, self.run)
        self.create_timer(self.DOB_dt, self.run_SSI)
    
    def clamp(self, value, min_value, max_value):
        """Clamp the value to the range [min_value, max_value]"""
        return max(min_value, min(value, max_value))

    def init_rff_parameters(self, M, dim_z, sigma_w=0.01):

        # w_i ~ N(0, sigma_w^2 I)
        w = np.random.normal(loc=0.0, scale=sigma_w, size=(M, dim_z))
        
        # b_i ~ Uniform(0, 2π)
        b = np.random.uniform(0.0, 2.0 * np.pi, size=(M,))
        
        return w, b


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


    def convert_thrust_to_pwm(self, rpm_thrust, thr):
        """Convert thrust level to PWM signal"""

        thr_new = rpm_thrust

        deadzone = 22.0*22.0
        threshold = 100.0

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


        dob_thrust = thrust

        if thrust < 0.0:
            pwm = 3.9 * thrust + 1450.0
            return self.clamp(pwm, 1000.0, 1450.0), thr, dob_thrust  # Any value <= 0 thrust maps to PWM 1000
        else:
            # Calculate PWM based on the thrust
            pwm = 3.9 * thrust + 1550.0

            return self.clamp(pwm, 1550.0, 2000.0), thr, dob_thrust  # Ensure PWM is within the bounds
    

    def ekf_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
        self.p = self.yaw_discontinuity(self.p)

        self.states = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.input_history[0][0], self.input_history[0][1]])


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
        # goal = ref
        # if ref > np.pi:
        #     while goal <= np.pi:
        #         goal = goal - 2*np.pi
        # elif ref < -np.pi:
        #     while goal >= -np.pi:
        #         goal = goal + 2*np.pi
        # return goal

 
    def run(self):
        k = self.k # -> 현재 시간을 index로 표시 -> 그래야 ref trajectory설정가능(******** todo ********)
                
        if time.time() - self.start_time > 1:          
            t = time.time()
            ##### Reference States ######
            for j in range(self.N+1):
                refs = self.reference[k+j,:]
                refs[2] = self.yaw_discontinuity(refs[2])
                yref = np.hstack((refs[0]-offset[0],refs[1]-offset[1],refs[2],refs[3],0,0,0,0,0,0))
                if j == self.N:
                    yref = np.hstack((refs[0]-offset[0],refs[1]-offset[1],refs[2],refs[3],0,0,0,0))
                self.ocp_solver.cost_set(j, "yref", yref)
                
                self.ref_check[j] = refs[0]-offset[0]
                # print(refs[3])
            # print(self.ref_check)
            obs_base = self.plot_traj_xy - offset

            # print(self.plot_traj_xy)
            # print(obs_base)

            ##### Obstacle Position ######
            # if self.DOB_switch:
            #     obs_pos = np.array([1000.0, +40.0, 6.0,  # Obstacle-1: x, y, radius
            #                         -0.0, +1000.0, 8.0,          
            #                         # 0.0,0.0,0.0]) # Obstacle-2: x, y, radius
            #                         self.param_filtered[0],self.param_filtered[1],self.param_filtered[2]]) # Obstacle-2: x, y, radius    
            # else:
            #     obs_pos = np.array([1000.0, +40.0, 6.0,  # Obstacle-1: x, y, radius
            #                         -0.0, +1000.0, 8.0,          
            #                         0.0,0.0,0.0]) # Obstacle-2: x, y, radius
            #                         # self.param_filtered[0],self.param_filtered[1],self.param_filtered[2]]) # Obstacle-2: x, y, radius
                
            if self.DOB_switch:
                # 1) alpha_u, alpha_v, alpha_r   (each M)
                alpha_part = np.hstack([self.alpha_u, self.alpha_v, self.alpha_r])

                # 2) w flattened (M*dim_z)
                w_part = self.w.flatten()

                # 3) b (M)
                b_part = self.b

                # Complete p vector
                param_vec = np.hstack([alpha_part, w_part, b_part])

            else:
                # When DOB_switch is OFF → use zeros
                total_dim = 3*self.M + self.M*self.dim_z + self.M
                param_vec = np.zeros(total_dim)

            # Send parameters to all shooting nodes
            for j in range(self.N+1):
                self.ocp_solver.set(j, "p", param_vec)
        
            # do stuff
            elapsed = time.time() - t
            # print(elapsed)
            
            
            # preparation phase
            self.ocp_solver.options_set('rti_phase', 1)
            status = self.ocp_solver.solve()
            t_preparation = self.ocp_solver.get_stats('time_tot')

            # set initial state
            ### State update ########################################
            if self.DOB_switch:
                disturbance = self.param_estim
            else:
                disturbance = np.array([0,0,0])
            
            pred_state = forward_prediction(self.states, self.input_history, self.delay, self.con_dt, disturbance)
            # print(pred_state)
            self.ocp_solver.set(0, "lbx", pred_state)
            self.ocp_solver.set(0, "ubx", pred_state)

            # feedback phase
            self.ocp_solver.options_set('rti_phase', 2)
            status = self.ocp_solver.solve()
            t_feedback = self.ocp_solver.get_stats('time_tot')

            # obtain mpc input
            x_con = self.ocp_solver.get(1, "x")
            del_con = self.ocp_solver.get(0, "u")
            # self.delta = x_con[6]
            # self.F = x_con[7]
            
            self.delta += del_con[0]*self.con_dt
            self.F += del_con[1]*self.con_dt


            new_input = np.array([self.delta, self.F])
            self.input_history = np.vstack([self.input_history[1:], new_input])
            # print(self.input_history[-1])
            

            self.now_delta = self.input_history[0][0]

            self.get_logger().info(f"MPC Computation Time: {t_preparation + t_feedback:.4f}s")


            # Publish the control inputs (e.g., thrust commands)
            
            self.delta_pwm = self.convert_steering_to_pwm(self.now_delta)
            # self.delta_pwm = self.convert_steering_to_pwm(self.delta)
            self.F_pwm, self.thr, self.dob_thrust = self.convert_thrust_to_pwm(self.F*100.0, self.thr)                        
            actuator_msg = Float64MultiArray()
            actuator_msg.data = [self.delta_pwm, self.F_pwm, 0.0, 0.0]
            # print(del_con[0], del_con[1])
            self.publisher_.publish(actuator_msg)        
            # print(self.delta_pwm, self.F_pwm)                        
            
            
            # Publish predicted states and reference states
            mpc_data_stack = MPCTraj()
            # mpc_data_stack.header.stamp = self.get_clock()
            mpc_data_stack.pred_num = float(self.N)
            mpc_data_stack.sampling_time = self.con_dt
            mpc_data_stack.cpu_time = t_preparation + t_feedback	
            mpc_data_stack.ref_num = 1000.0	
            mpc_data_stack.ref_dt = self.ref_dt	
            mpc_data_stack.traj_x = self.plot_traj_xy[0]	
            mpc_data_stack.traj_y = self.plot_traj_xy[1]
            mpc_data_stack.theta = self.theta	
            mpc_data_stack.a = self.A	
            mpc_data_stack.b = self.B	
            mpc_data_stack.c = self.C	
            
            for j in range(self.N+1):
                mpc_pred = MPCState()
                mpc_ref = MPCState()
                mpc_pred.x = self.ocp_solver.get(j, "x")[0]+offset[0]
                mpc_pred.y = self.ocp_solver.get(j, "x")[1]+offset[1]
                mpc_pred.p = self.ocp_solver.get(j, "x")[2]
                mpc_pred.u = self.ocp_solver.get(j, "x")[3]
                mpc_pred.v = self.ocp_solver.get(j, "x")[4]
                mpc_pred.r = self.ocp_solver.get(j, "x")[5]
                mpc_pred.delta = self.ocp_solver.get(j, "x")[6]
                mpc_pred.f = self.ocp_solver.get(j, "x")[7]
                mpc_data_stack.state.append(mpc_pred)            
                # print(mpc_pred.u)
                mpc_ref.x = self.reference[k+j,0]
                mpc_ref.y = self.reference[k+j,1]
                mpc_ref.p = self.reference[k+j,2]
                mpc_ref.u = self.reference[k+j,3]
                mpc_ref.v = 0.0
                mpc_ref.r = self.reference[k+j,4]
                mpc_ref.delta = 0.0
                mpc_ref.f = 0.0
                mpc_data_stack.ref.append(mpc_ref)            


            obs_state = ObsState()
            obs_state.x   = 0.0 +offset[0]
            obs_state.y   = 0.0 +offset[1]
            obs_state.rad = 0.0 
            mpc_data_stack.obs.append(obs_state)
            obs_state = ObsState()
            obs_state.x   = 0.0 +offset[0]
            obs_state.y   = 0.0 +offset[1]
            obs_state.rad = 0.0
            mpc_data_stack.obs.append(obs_state)        

            self.mpcvis_pub.publish(mpc_data_stack)
            
            # Increment the index for the reference trajectory
            self.k += 1
            if self.k + self.N >= len(self.reference):
                self.k = 0  # Reset the index if it goes beyond the reference length

        else:
            # print(self.states[0])
            # print(self.states[0] + offset[0])
            self.plot_traj_xy = (self.states[0] + offset[0], self.states[1] + offset[1])
            
            # self.reference = generate_figure_eight_trajectory_con(1000, self.ref_dt, self.plot_traj_xy, self.theta, self.A, self.B, self.C) # reference dt = 0.01 sec, 1000 sec trajectory generation
            self.reference = generate_figure_eight_trajectory(1000, self.ref_dt, self.plot_traj_xy, self.theta, self.A, self.B, self.C) # reference dt = 0.01 sec, 1000 sec trajectory generation
            
            self.reference = self.reference[::self.ref_iter,:]


            print("prepare time : ",time.time() - self.start_time)
            self.ocp_solver.options_set('rti_phase', 1)
            status = self.ocp_solver.solve()
            t_preparation = self.ocp_solver.get_stats('time_tot')
            
            # set initial state
            ### State update ###
            if self.DOB_switch:
                disturbance = self.param_estim
            else:
                disturbance = np.array([0,0,0])
            
            pred_state = forward_prediction(self.states, self.input_history, self.delay, self.con_dt, disturbance)
            self.ocp_solver.set(0, "lbx", pred_state)
            self.ocp_solver.set(0, "ubx", pred_state)

    

    def run_SSI(self):
        # dob_state = self.states
        
        # self.state_estim, self.param_estim, self.param_filtered = DOB(
        #     self.input_history[0], dob_state, self.state_estim, self.param_filtered, self.param_estim, self.DOB_dt
        # )
        # DOB_msg = Float64MultiArray()
        # DOB_msg.data = [self.param_filtered[0], self.param_filtered[1], self.param_filtered[2]]
        # self.DOB_pub.publish(DOB_msg)

        # ---------------------------
        # 0. Input extraction
        # ---------------------------
        delta = self.input_history[0][0]
        F = self.input_history[0][1]

        # ---------------------------
        # 1. Form z vector
        # ---------------------------
        z = np.array([self.u, self.v, self.r, delta, F])

        # ---------------------------
        # 2. RFF feature vector
        # ---------------------------
        phi = np.cos(self.w @ z + self.b)    # shape = (M,)

        # ---------------------------
        # 3. nominal model f_nom (u̇, v̇, ṙ)
        # ---------------------------    
        eps = 0.0001

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

        # thrust model
        s = 25
        k = 8
        a1 = 2.2 * 2.2
        a2 = 2.2 * 2.2
        b11 = 1.0
        b22 = 1.0 

        T = ((1/(1+exp(s*F)))*(b11*F + tanh(k*F)*a1) +
            (1/(1+exp(-s*F)))*(b22*F + tanh(k*F)*a2))

        f_nom = np.array([
            # u_dot
            ( - Xu*self.u
            - Xuu * sqrt(self.u*self.u + eps) * self.u
            + alpha1 * T * cos(alpha2*delta) ),

            # v_dot
            ( -Yv*self.v
            -Yr*self.r
            -Yvv * sqrt(self.v*self.v + eps) * self.v
            + alpha3 * T * sin(alpha2*delta) ),

            # r_dot
            ( -Nr*self.r
            -Nv*self.v
            -Nrr * sqrt(self.r*self.r + eps) * self.r
            -alpha4*T*sin(alpha2*delta) )
        ])

        # ---------------------------
        # 4. actual derivatives (u̇, v̇, ṙ)
        # ---------------------------
        # dx_actual = (previous - current)/dt  → sign 바꾸고 싶으면 변경 가능
        dx_actual = (self.states_prev - self.states) / self.DOB_dt

        du_actual = dx_actual[3]
        dv_actual = dx_actual[4]
        dr_actual = dx_actual[5]

        # ---------------------------
        # 5. residuals
        # ---------------------------
        res_u = du_actual - f_nom[0]
        res_v = dv_actual - f_nom[1]
        res_r = dr_actual - f_nom[2]

        # ---------------------------
        # 6. h_hat(z) per dimension
        # ---------------------------
        h_u = (self.alpha_u @ phi) / self.M
        h_v = (self.alpha_v @ phi) / self.M
        h_r = (self.alpha_r @ phi) / self.M

        # ---------------------------
        # 7. α update (3 sets)
        # ---------------------------
        lr = 0.01

        self.alpha_u = self.alpha_u - lr * (h_u - res_u) * phi
        self.alpha_v = self.alpha_v - lr * (h_v - res_v) * phi
        self.alpha_r = self.alpha_r - lr * (h_r - res_r) * phi

        # ---------------------------
        # 8. Save states
        # ---------------------------
        self.states_prev = self.states.copy()

        # ---------------------------
        # 9. Publish disturbance estimate
        # ---------------------------
        msg = Float64MultiArray()
        msg.data = [h_u, h_v, h_r]
        self.DOB_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = AuraMPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()