from acados_setting_nlp1 import *
from acados_setting_nlp2 import *
from bow_CIA_pycombina import *
from sample_object import *
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
        self.delta = self.F = self.F_eff = 0.0
        self.delta_pwm = self.F_pwm = 1500.0
        self.states = np.zeros(8)
        self.states2 = np.zeros(7)
        self.thr = 0.0
        self.del_thr_max = 0.5
        self.dob_thrust = 0.0
        self.stop_switch = 0.0
        self.bow_thrust = 0.0


        self.F = 1.0
        
        # MPC parameter settings
        self.Tf = 20 # prediction time 4 sec
        self.N = 40 # prediction horizon
        self.con_dt = 0.5 # control sampling time
        
        self.init_history = np.zeros([self.N, 8])

        # inputs  = vertcat(delta_d, F_d, B)
        

        Q_mat1 = 1*np.diag([1e1, 1e1, 1e3, 1e4, 1e4, 1e-2, 1e-1, 1e0])
        Q_mat_terminal1 = 1*np.diag([1e4, 1e4, 1e6, 1e3, 1e3, 1e-2, 1e-1, 1e0])
        R_mat1 = 1*np.diag([1e1, 1e-1, 1e4])

        Q_mat2 = 1*np.diag([1e1, 1e1, 1e3, 1e3, 1e3, 1e-2, 1e0])
        Q_mat_terminal2 = 1*np.diag([1e4, 1e4, 1e6, 1e4, 1e4, 1e-2, 1e0])
        R_mat2 = 1*np.diag([1e1])    

        self.state_history = np.zeros([self.N, 8])

        self.ocp_solver_nlp1 = setup_trajectory_tracking_nlp1(self.states, self.N, self.Tf, Q_mat1, Q_mat_terminal1, R_mat1)
        self.ocp_solver_nlp2 = setup_trajectory_tracking_nlp2(self.states2, self.N, self.Tf, Q_mat2, Q_mat_terminal2, R_mat2)


        # DOB
        # DOB(state, state_estim, param_filtered, param_estim, dt):
        self.state_estim = np.array([0.0, 0.0, 0.0])
        self.param_filtered = np.array([0.0, 0.0, 0.0])
        self.param_estim = np.array([0.0, 0.0, 0.0])
        self.DOB_dt = 0.1

        self.a_dot_state = 0.0

        self.before_thrust = 0.0
 
        #CIA
        self.count_on_left = 0
        self.count_on_right = 0
        self.count_on_zero = 10
        self.gear_before = 0
        self.gear_switch = 0.0

        self.bow_count_on_left = 0
        self.bow_count_on_right = 0
        self.bow_count_on_zero = 10
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

    def convert_thrust_to_pwm(self, rpm_thrust, thr, before):
        """Convert thrust level to PWM signal"""
        # thr_new = np.sign(rpm_thrust - thr)*100*0.5 + thr

        # if (rpm_thrust-thr)*(rpm_thrust-thr_new)<0:
        #     thr_new = rpm_thrust

        if rpm_thrust > 0.5:
            thrust = 22.0
        elif rpm_thrust < -0.5:
            thrust = -22.0
        else:
            thrust = 0.0

        
        dob_thrust = thrust

        if thrust < -0.1:
            pwm = 3.9 * thrust + 1450.0
            return self.clamp(pwm, 1000.0, 1450.0), thr, dob_thrust  # Any value <= 0 thrust maps to PWM 1000
        elif thrust > 0.1:
            pwm = 3.9 * thrust + 1550.0
            return self.clamp(pwm, 1550.0, 2000.0), thr, dob_thrust  # Ensure PWM is within the bounds
        else:
            if before >= 0:
                pwm = 1550.0
                return pwm, thr, dob_thrust
            else:
                pwm = 1450.0
                return pwm, thr, dob_thrust

    def start_int_scalar(self, mean, var):
        rng = np.random.default_rng()
        x = rng.normal(loc=mean, scale=np.sqrt(var))
        return int(round(x))

    def kappa_int_scalar(self, mean, var, dwell):
        rng = np.random.default_rng()
        x = np.clip(rng.normal(loc=mean, scale=np.sqrt(var)), dwell, self.N)
        return int(round(x))

    def ekf_callback(self, msg):# - frequency = gps callback freq. 
        """Callback to update states from EKF estimated state."""
        self.x, self.y, self.p, self.u, self.v, self.r = msg.data[:6]
        self.states = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.delta, self.F])
        self.states2 = np.array([self.x-offset[0], self.y-offset[1], self.p, self.u, self.v, self.r, self.delta])
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

    def reconstruct_sequence(self, switch_history, starts, kappas, N):
        """
        Build a ternary sequence of length N from starts/kappas and modes in switch_history.
        Each switch i fills [start_i, start_i + kappa_i) with mode_i (±1). Elsewhere stays 0.
        """
        seq = np.zeros(N, dtype=int)
        L = len(switch_history)
        for i in range(L):
            s = int(starts[i])
            k = int(kappas[i])
            m = int(switch_history[i][2])  # mode: ±1
            if k <= 0:
                continue
            s0 = max(0, min(s, N-1))
            e0 = max(s0, min(s + k, N))
            if e0 > s0:
                seq[s0:e0] = m
        return seq

 
    def _sample_channel(self, switch_history, N, cov, dwell, rng, max_retries=10):
        """
        Sampling with 'lock-first-start-if-zero':
        - If switch_history[0][0] == 0, we fix starts[0] = 0 (never changed) and only sample its kappa.
        - Other starts are sampled and clamped into [1, N-1], then enforced strictly increasing.
        - Dwell applies except tail-relax for the last start in [N - dwell, N).
        """
        L = len(switch_history)
        if L == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        # whether the very first run is anchored at t=0
        lock_first_zero = (switch_history[0][0] == 0)

        for _ in range(max_retries):
            # 1) draw starts
            starts = np.zeros(L, dtype=int)

            for i in range(L):
                if i == 0 and lock_first_zero:
                    # lock the first start at 0, do NOT move it
                    s = 0
                else:
                    # sample around mean and clamp to [1, N-1]
                    s = int(round(rng.normal(loc=switch_history[i][0], scale=np.sqrt(cov))))
                    s = max(1, min(s, N-1))
                starts[i] = s

            # enforce strictly increasing for i >= 1 (do not touch starts[0] even if it's 0)
            for i in range(1, L):
                if starts[i] <= starts[i-1]:
                    starts[i] = starts[i-1] + 1
                    if starts[i] > N-1:
                        break
            if starts[-1] > N-1:
                continue

            # 2) draw kappas with bounds
            kappas = np.zeros(L, dtype=int)
            feasible = True
            for i in range(L):
                if i < L-1:
                    max_kappa_by_next = max(0, starts[i+1] - starts[i])
                    min_req = dwell  # normal dwell applies for non-last
                    tail_relax = False
                else:
                    max_kappa_by_next = max(0, (N-1) - starts[i])
                    # Relax dwell if the last switch starts in [N - dwell, N)
                    tail_relax = (starts[i] >= (N - dwell))
                    min_req = 0 if tail_relax else dwell

                # If even the max available is smaller than the required min (unless tail relax), infeasible
                if (not tail_relax) and (max_kappa_by_next < min_req):
                    feasible = False
                    break

                # sample around mean and clip into [min_req, max_kappa_by_next]
                k_mean = switch_history[i][1]
                k = int(round(rng.normal(loc=k_mean, scale=np.sqrt(cov))))
                k = max(min_req, min(k, max_kappa_by_next))
                kappas[i] = k

            if not feasible:
                continue

            # 3) final overlap/end checks
            ok = True
            for i in range(L-1):
                if starts[i] + kappas[i] > starts[i+1]:
                    ok = False
                    break
            if ok and (starts[-1] + kappas[-1] <= (N-1)):
                return starts, kappas
            # (if tail_relax kicked in, the last inequality already respects the cap)

        return None, None



    def run(self):
        k = self.k # 
        # t = time.time()             

        for j in range(self.N+1):
            dock_x = 10.0
            dock_y = 10.0
            dock_psi = 0.0*3.141592/180

            dock_psi = self.yaw_discontinuity(dock_psi)
            yref1 = np.hstack((dock_x,dock_y,dock_psi,0,0,0,0,0,0,0,0))
            yref2 = np.hstack((dock_x,dock_y,dock_psi,0,0,0,0,0))
            if j == self.N:
                yref1 = np.hstack((dock_x,dock_y,dock_psi,0,0,0,0,0))
                yref2 = np.hstack((dock_x,dock_y,dock_psi,0,0,0,0))
            self.ocp_solver_nlp1.cost_set(j, "yref", yref1)
            self.ocp_solver_nlp2.cost_set(j, "yref", yref2)


        #################################### 1st NLP ####################################
        
        ##### Obstacle Position ######
        obs_pos = np.array([dock_x, dock_y, dock_psi]) # Obstacle-2: x, y, radius

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

        for j in range(self.N):
            self.state_history[j] = self.ocp_solver_nlp1.get(j,"x")

        # obtain mpc input
        gear_array = np.zeros(self.N)
        for j in range(self.N):
            if j == self.N-1:
                con = self.ocp_solver_nlp1.get(self.N,"x")
                gear_array[j] = con[7]
            else:
                con = self.ocp_solver_nlp1.get(j+1,"x")
                gear_array[j] = con[7]


        bow_array = np.zeros(self.N)
        for j in range(self.N):
            con = self.ocp_solver_nlp1.get(0,"u")
            bow_array[j] = con[2]

        bow_array = np.clip(bow_array, -1, 1)
        gear_array = np.clip(gear_array, -1, 1)

        #################################### Thruster CIA ####################################
        t = time.time()  

        dwell_time = 2.0
        stop_dwell_time = 2.0
                

        ### CIA process
        dwell_len = int(dwell_time / self.con_dt)
        stop_dwell_len = int(stop_dwell_time / self.con_dt)
        
        if self.count_on_left <= dwell_len - 1 and self.gear_switch > 0.1:
            self.count_on_left += 1
            start_idx = dwell_len - self.count_on_left 
            input_gear_array = bow_mapping(gear_array[start_idx:], self.con_dt, len(gear_array[start_idx:]),
                                        dwell_time, stop_dwell_time)
            CIA_gear_array = np.concatenate([np.ones(start_idx), input_gear_array])

            self.count_on_right = 0
            self.count_on_zero = 0


        elif self.count_on_right <= dwell_len - 1 and self.gear_switch < -0.1:
            self.count_on_right += 1
            start_idx = dwell_len - self.count_on_right 
            input_gear_array = bow_mapping(gear_array[start_idx:], self.con_dt, len(gear_array[start_idx:]),
                                        dwell_time, stop_dwell_time)
            CIA_gear_array = np.concatenate([-np.ones(start_idx), input_gear_array])

            self.count_on_left = 0
            self.count_on_zero = 0


        elif self.count_on_zero <= stop_dwell_len - 1 and self.gear_switch == 0.0:
            self.count_on_zero += 1
            start_idx = stop_dwell_len - self.count_on_zero 
            input_gear_array = bow_mapping(gear_array[start_idx:], self.con_dt, len(gear_array[start_idx:]),
                                        dwell_time, stop_dwell_time)
            CIA_gear_array = np.concatenate([np.zeros(start_idx), input_gear_array])

            self.count_on_left = 0
            self.count_on_right = 0


        elif self.count_on_right >= dwell_len and self.gear_switch > 0.1:
            self.count_on_right = 0
            CIA_gear_array = bow_mapping(gear_array, self.con_dt, len(gear_array), dwell_time, stop_dwell_time)
        

        elif self.count_on_left >= dwell_len and self.gear_switch < -0.1:
            self.count_on_left = 0
            CIA_gear_array = bow_mapping(gear_array, self.con_dt, len(gear_array), dwell_time, stop_dwell_time)


        elif self.count_on_zero >= stop_dwell_len and self.gear_switch != 0.0:
            self.count_on_zero = 0
            CIA_gear_array = bow_mapping(gear_array, self.con_dt, len(gear_array), dwell_time, stop_dwell_time)


        else:
            CIA_gear_array = bow_mapping(gear_array, self.con_dt, len(gear_array), dwell_time, stop_dwell_time)

    

        #################################### Bow CIA ####################################
        
        bow_dwell_time = 2.0
        bow_stop_dwell_time = 2.0
                
        ### CIA process
        bow_dwell_len = int(bow_dwell_time / self.con_dt)
        bow_stop_dwell_len = int(bow_stop_dwell_time / self.con_dt)
        
        if self.bow_count_on_left <= bow_dwell_len - 1 and self.bow_switch > 0.1:
            self.bow_count_on_left += 1
            start_idx = bow_dwell_len - self.bow_count_on_left 
            input_bow_array = bow_mapping(bow_array[start_idx:], self.con_dt, len(bow_array[start_idx:]),
                                        bow_dwell_time, bow_stop_dwell_time)
            CIA_bow_array = np.concatenate([np.ones(start_idx), input_bow_array])

            self.bow_count_on_right = 0
            self.bow_count_on_zero = 0


        elif self.bow_count_on_right <= bow_dwell_len - 1 and self.bow_switch < -0.1:
            self.bow_count_on_right += 1
            start_idx = bow_dwell_len - self.bow_count_on_right 
            input_bow_array = bow_mapping(bow_array[start_idx:], self.con_dt, len(bow_array[start_idx:]),
                                        bow_dwell_time, bow_stop_dwell_time)
            CIA_bow_array = np.concatenate([-np.ones(start_idx), input_bow_array])

            self.bow_count_on_left = 0
            self.bow_count_on_zero = 0


        elif self.bow_count_on_zero <= bow_stop_dwell_len - 1 and self.bow_switch == 0.0:
            self.bow_count_on_zero += 1
            start_idx = bow_stop_dwell_len - self.bow_count_on_zero 
            input_bow_array = bow_mapping(bow_array[start_idx:], self.con_dt, len(bow_array[start_idx:]),
                                        bow_dwell_time, bow_stop_dwell_time)
            CIA_bow_array = np.concatenate([np.zeros(start_idx), input_bow_array])

            self.bow_count_on_left = 0
            self.bow_count_on_right = 0


        elif self.bow_count_on_right >= bow_dwell_len and self.bow_switch > 0.1:
            self.bow_count_on_right = 0
            CIA_bow_array = bow_mapping(bow_array, self.con_dt, len(bow_array), bow_dwell_time, bow_stop_dwell_time)
        

        elif self.bow_count_on_left >= bow_dwell_len and self.bow_switch < -0.1:
            self.bow_count_on_left = 0
            CIA_bow_array = bow_mapping(bow_array, self.con_dt, len(bow_array), bow_dwell_time, bow_stop_dwell_time)


        elif self.bow_count_on_zero >= bow_stop_dwell_len and self.bow_switch != 0.0:
            self.bow_count_on_zero = 0
            CIA_bow_array = bow_mapping(bow_array, self.con_dt, len(bow_array), bow_dwell_time, bow_stop_dwell_time)

        else:
            CIA_bow_array = bow_mapping(bow_array, self.con_dt, len(bow_array), bow_dwell_time, bow_stop_dwell_time)


        #################################### Switch time replanning ####################################

        ############## Swith check ##############
        # Gear
        gear_switch_history = []

        for j in range(self.N-1):
            if CIA_gear_array[j] == 0 and CIA_gear_array[j+1] == 1:
                jj = 1
                kappa = 0
                while CIA_gear_array[j+jj] == 1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                gear_switch_history.append([j+1, kappa, 1]) 

            elif CIA_gear_array[j] == 0 and CIA_gear_array[j+1] == -1:
                jj = 1
                kappa = 0
                while CIA_gear_array[j+jj] == -1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                gear_switch_history.append([j+1, kappa, -1]) 

            elif CIA_gear_array[j] == 1 and CIA_gear_array[j+1] == -1:
                jj = 1
                kappa = 0
                while CIA_gear_array[j+jj] == -1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                gear_switch_history.append([j+1, kappa, -1]) 

            elif CIA_gear_array[j] == -1 and CIA_gear_array[j+1] == 1:
                jj = 1
                kappa = 0
                while CIA_gear_array[j+jj] == 1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                gear_switch_history.append([j+1, kappa, 1]) 

            elif CIA_gear_array[0] == 1:
                jj = 1
                kappa = 0
                while CIA_gear_array[j+jj] == 1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                gear_switch_history.append([0, kappa, 1]) 

            elif CIA_gear_array[0] == -1:
                jj = 1
                kappa = 0
                while CIA_gear_array[j+jj] == -1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                gear_switch_history.append([0, kappa, -1])                 
           


        # bow
        bow_switch_history = []

        for j in range(self.N-1):
            if CIA_bow_array[j] == 0 and CIA_bow_array[j+1] == 1:
                jj = 1
                kappa = 0
                while CIA_bow_array[j+jj]  == 1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                bow_switch_history.append([j+1, kappa, 1]) 

            elif CIA_bow_array[j] == 0 and CIA_bow_array[j+1] == -1:
                jj = 1
                kappa = 0
                while CIA_bow_array[j+jj] == -1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                bow_switch_history.append([j+1, kappa, -1]) 

            elif CIA_bow_array[j] == 1 and CIA_bow_array[j+1] == -1:
                jj = 1
                kappa = 0
                while CIA_bow_array[j+jj] == -1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                bow_switch_history.append([j+1, kappa, -1]) 

            elif CIA_bow_array[j] == -1 and CIA_bow_array[j+1] == 1:
                jj = 1
                kappa = 0
                while CIA_bow_array[j+jj] == 1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                bow_switch_history.append([j+1, kappa, 1]) 

            elif CIA_bow_array[0] == 1:
                jj = 1
                kappa = 0
                while CIA_bow_array[j+jj] == 1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                bow_switch_history.append([0, kappa, 1]) 

            elif CIA_bow_array[0] == -1:
                jj = 1
                kappa = 0
                while CIA_bow_array[j+jj] == -1 and j+jj < self.N-1:
                    kappa += 1
                    jj += 1
                bow_switch_history.append([0, kappa, -1])                 
           


        # print(gear_switch_history)

        ############## Sampling ##############
        # Assumptions:
        # - self.N: horizon length (number of discrete steps)
        # - start_int_scalar(mean, var), kappa_int_scalar(mean, var, dwell): given
        # - obj_calc(CIA_gear_array, CIA_bow_array, self.state_history): given
        # ===== Helper: build a ternary sequence from (start,kappa,mode) =====

        ############## Sampling (objective uses NEW sequences) ##############
        # samples = 100
        # cov = 1.0
        # dwell = 4
        # rng = np.random.default_rng()

        # gear_len = len(gear_switch_history)
        # bow_len  = len(bow_switch_history)

        # # columns: [gear(s,k)*G, bow(s,k)*B, obj]
        # num_cols = 2*gear_len + 2*bow_len + 1
        # sample_history = np.zeros((samples, num_cols), dtype=float)


        # for s in range(samples):
        #     # sample gear
        #     gear_starts, gear_kappas = self._sample_channel(
        #         gear_switch_history, self.N, cov, dwell, rng
        #     )
        #     if gear_starts is None:
        #         sample_history[s, -1] = -1
        #         continue

        #     # sample bow
        #     bow_starts, bow_kappas = self._sample_channel(
        #         bow_switch_history, self.N, cov, dwell, rng
        #     )
        #     if bow_starts is None:
        #         sample_history[s, -1] = -1
        #         continue

        #     # reconstruct NEW sequences from sampled schedules
        #     gear_seq = self.reconstruct_sequence(gear_switch_history, gear_starts, gear_kappas, self.N)
        #     bow_seq  = self.reconstruct_sequence(bow_switch_history,  bow_starts,  bow_kappas,  self.N)

        #     # compute objective on the NEW sequences
        #     obj = obj_calc(gear_seq, bow_seq, self.state_history,self.N)

        #     # pack into sample_history
        #     cursor = 0
        #     for j in range(gear_len):
        #         sample_history[s, cursor] = gear_starts[j]; cursor += 1
        #         sample_history[s, cursor] = gear_kappas[j]; cursor += 1
        #     for j in range(bow_len):
        #         sample_history[s, cursor] = bow_starts[j];  cursor += 1
        #         sample_history[s, cursor] = bow_kappas[j];  cursor += 1
        #     sample_history[s, cursor] = obj


        # # ===== Pick best row and return final sequences computed from NEW schedules =====
        # valid_mask = sample_history[:, -1] != -1
        # if not np.any(valid_mask):
        #     raise RuntimeError("No feasible samples found (all objectives == -1).")

        # valid_idx = np.where(valid_mask)[0]
        # best_idx_local = np.argmin(sample_history[valid_mask, -1])
        # best_row_idx   = valid_idx[best_idx_local]
        # best_obj       = sample_history[best_row_idx, -1]

        # # unpack starts/kappas
        # cursor = 0
        # gear_starts = sample_history[best_row_idx, cursor : cursor + 2*gear_len : 2].astype(int)
        # gear_kappas = sample_history[best_row_idx, cursor+1: cursor + 2*gear_len : 2].astype(int)
        # cursor += 2 * gear_len
        # bow_starts  = sample_history[best_row_idx, cursor : cursor + 2*bow_len : 2].astype(int)
        # bow_kappas  = sample_history[best_row_idx, cursor+1: cursor + 2*bow_len : 2].astype(int)

        # # reconstruct sequences for the winner
        # best_gear_seq = self.reconstruct_sequence(gear_switch_history, gear_starts, gear_kappas, self.N)
        # best_bow_seq  = self.reconstruct_sequence(bow_switch_history,  bow_starts,  bow_kappas,  self.N)

        # print("original : ",CIA_bow_array)
        # print("optimal : ",best_bow_seq)
        # origin_obj = obj_calc(CIA_gear_array, CIA_bow_array, self.state_history, self.N)
        # print(f"Original objective={origin_obj}, optimal objective={best_obj}")
        #################################### Switch time replanning ####################################

        # ... your switch_history builders above ...

        samples = 50
        cov = 1.0
        dwell = 4
        rng = np.random.default_rng()

        gear_len = len(gear_switch_history)
        bow_len  = len(bow_switch_history)

        # === Case A: no switches at all → skip sampling, keep originals ===
        if gear_len == 0 and bow_len == 0:
            best_gear_seq = CIA_gear_array.copy()
            best_bow_seq  = CIA_bow_array.copy()
            origin_obj = obj_calc(CIA_gear_array, CIA_bow_array, self.state_history, self.N)
            print("original : ", CIA_gear_array)
            print("optimal  : ", best_gear_seq)
            print(f"Original objective={origin_obj}, optimal objective={origin_obj}")
            # if this is inside a function/method and you want to return here, do so.
        else:
            # === Case B: at least one channel has switches → sample what exists ===
            num_cols = 2*gear_len + 2*bow_len + 1
            sample_history = np.zeros((samples, num_cols), dtype=float)

            for s in range(samples):
                # ---- Gear sampling or passthrough ----
                if gear_len == 0:
                    gear_starts = np.array([], dtype=int)
                    gear_kappas = np.array([], dtype=int)
                    gear_seq = CIA_gear_array.copy()  # no switches → keep original gear sequence
                else:
                    gear_starts, gear_kappas = self._sample_channel(
                        gear_switch_history, self.N, cov, dwell, rng
                    )
                    if gear_starts is None:
                        sample_history[s, -1] = -1
                        continue
                    gear_seq = self.reconstruct_sequence(gear_switch_history, gear_starts, gear_kappas, self.N)

                # ---- Bow sampling or passthrough ----
                if bow_len == 0:
                    bow_starts = np.array([], dtype=int)
                    bow_kappas = np.array([], dtype=int)
                    bow_seq = CIA_bow_array.copy()  # no switches → keep original bow sequence
                else:
                    bow_starts, bow_kappas = self._sample_channel(
                        bow_switch_history, self.N, cov, dwell, rng
                    )
                    if bow_starts is None:
                        sample_history[s, -1] = -1
                        continue
                    bow_seq = self.reconstruct_sequence(bow_switch_history, bow_starts, bow_kappas, self.N)

                # ---- Objective on NEW sequences (with passthrough where needed) ----
                obj = obj_calc(gear_seq, bow_seq, self.state_history, self.N)

                # ---- Pack starts/kappas (skip the empty side(s)) ----
                cursor = 0
                if gear_len > 0:
                    for j in range(gear_len):
                        sample_history[s, cursor] = gear_starts[j]; cursor += 1
                        sample_history[s, cursor] = gear_kappas[j]; cursor += 1
                if bow_len > 0:
                    for j in range(bow_len):
                        sample_history[s, cursor] = bow_starts[j];  cursor += 1
                        sample_history[s, cursor] = bow_kappas[j];  cursor += 1
                sample_history[s, cursor] = obj

            # ===== Pick best row (among feasible) =====
            valid_mask = sample_history[:, -1] != -1
            if not np.any(valid_mask):
                # Fallback: nothing feasible → keep originals
                best_gear_seq = CIA_gear_array.copy()
                best_bow_seq  = CIA_bow_array.copy()
                best_obj = obj_calc(best_gear_seq, best_bow_seq, self.state_history, self.N)
            else:
                valid_idx = np.where(valid_mask)[0]
                best_idx_local = np.argmin(sample_history[valid_mask, -1])
                best_row_idx   = valid_idx[best_idx_local]
                best_obj       = sample_history[best_row_idx, -1]

                # ---- Unpack starts/kappas conditionally ----
                cursor = 0
                if gear_len > 0:
                    gear_starts = sample_history[best_row_idx, cursor : cursor + 2*gear_len : 2].astype(int)
                    gear_kappas = sample_history[best_row_idx, cursor+1: cursor + 2*gear_len : 2].astype(int)
                    cursor += 2*gear_len
                else:
                    gear_starts = np.array([], dtype=int)
                    gear_kappas = np.array([], dtype=int)

                if bow_len > 0:
                    bow_starts = sample_history[best_row_idx, cursor : cursor + 2*bow_len : 2].astype(int)
                    bow_kappas = sample_history[best_row_idx, cursor+1: cursor + 2*bow_len : 2].astype(int)
                else:
                    bow_starts = np.array([], dtype=int)
                    bow_kappas = np.array([], dtype=int)

                # ---- Reconstruct winner (passthrough where empty) ----
                if gear_len > 0:
                    best_gear_seq = self.reconstruct_sequence(gear_switch_history, gear_starts, gear_kappas, self.N)
                else:
                    best_gear_seq = CIA_gear_array.copy()

                if bow_len > 0:
                    best_bow_seq  = self.reconstruct_sequence(bow_switch_history,  bow_starts,  bow_kappas,  self.N)
                else:
                    best_bow_seq  = CIA_bow_array.copy()

            origin_obj = obj_calc(CIA_gear_array, CIA_bow_array, self.state_history, self.N)
            if origin_obj <= best_obj:
                best_gear_seq = CIA_gear_array.copy()
                best_bow_seq  = CIA_bow_array.copy()
                best_obj      = origin_obj

            print("original : ", CIA_gear_array)
            print("optimal  : ", best_gear_seq)
            print(f"Original objective={origin_obj}, optimal objective={best_obj}")
            print(f"Delta objective={origin_obj - best_obj}")



        #################################### 2nd NLP ####################################
        
        ##### Obstacle Position ######
        for j in range(self.N):
            obs_pos = np.array([dock_x, dock_y, dock_psi, best_gear_seq[j], best_bow_seq[j]]) 
            self.ocp_solver_nlp2.set(j, "p", obs_pos)
    

        # preparation phase
        self.ocp_solver_nlp2.options_set('rti_phase', 1)
        status = self.ocp_solver_nlp2.solve()
        t_preparation = self.ocp_solver_nlp2.get_stats('time_tot')

        # set initial state
        self.ocp_solver_nlp2.set(0, "lbx", self.states2)
        self.ocp_solver_nlp2.set(0, "ubx", self.states2)

        # feedback phase
        self.ocp_solver_nlp2.options_set('rti_phase', 2)
        status = self.ocp_solver_nlp2.solve()
        t_feedback = self.ocp_solver_nlp2.get_stats('time_tot')

        # obtain mpc input
        del_con = self.ocp_solver_nlp2.get(0, "u")
        self.delta += del_con[0]*self.con_dt
        self.F = float(best_gear_seq[0])
        self.gear_switch = float(best_gear_seq[0])

        self.bow_thrust = float(best_bow_seq[0])
        self.bow_switch = float(best_bow_seq[0])
        # print(best_bow_seq)

        # self.get_logger().info(f"MPC Computation Time: {t_preparation + t_feedback:.4f}s")


        self.delta_pwm = self.convert_steering_to_pwm(self.delta)
        self.F_pwm, self.thr, self.dob_thrust = self.convert_thrust_to_pwm(self.F, self.thr, self.before_thrust)                        
        actuator_msg = Float64MultiArray()
        actuator_msg.data = [self.delta_pwm, self.F_pwm, float(self.bow_switch), 0.0]


        self.publisher_.publish(actuator_msg)    

        self.before_thrust = self.F                            
        
        print("Time : ",time.time()-t)

        # Publish predicted states and reference states
        mpc_data_stack = MPCTraj()
        # mpc_data_stack.header.stamp = self.get_clock()
        mpc_data_stack.pred_num = float(self.N)
        mpc_data_stack.sampling_time = self.con_dt
        mpc_data_stack.cpu_time = t_preparation + t_feedback	
        mpc_data_stack.ref_num = 0.0	
        mpc_data_stack.ref_dt = 100.0	
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
            mpc_pred.f = self.ocp_solver_nlp2.get(j, "x")[6]
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
        obs_state.x   = obs_pos[0]+offset[0]
        obs_state.y   = obs_pos[1]+offset[1]
        obs_state.rad = obs_pos[2]
        mpc_data_stack.obs.append(obs_state)
        obs_state = ObsState()
        obs_state.x   = obs_pos[0]
        obs_state.y   = obs_pos[1]
        obs_state.rad = obs_pos[2]
        mpc_data_stack.obs.append(obs_state)        

        self.mpcvis_pub.publish(mpc_data_stack)
        


    def run_DOB(self):
        dob_state = self.states
        dob_state[7] = self.dob_thrust
        self.state_estim, self.param_estim, self.param_filtered = DOB(self.states, self.state_estim, self.param_filtered, self.param_estim, self.DOB_dt)
             
        DOB_msg = Float64MultiArray()
        DOB_msg.data = [0.2*self.bow_switch, self.param_filtered[1], self.param_filtered[2]]
        self.DOB_pub.publish(DOB_msg)                     
        


def main(args=None):
    rclpy.init(args=args)
    node = AuraMPC()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()