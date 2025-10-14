import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
import numpy as np
import pyproj


class ShipSimulator(Node):
    def __init__(self):
        super().__init__('ship_simulator')
        
        # ROS2 I/O
        self.state_pub = self.create_publisher(Float64MultiArray, '/ekf/estimated_state', 10)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', 10)
        self.control_sub = self.create_subscription(Float64MultiArray, '/actuator_outputs', self.control_callback, 10)

        # 초기 상태 [x, y, ψ, u, v, r, δ, F_eff]
        self.ship_state = np.zeros(6)
        self.ship_state[0] = (289577.66 + 291591.05) * 0.5  # X
        self.ship_state[1] = (4117065.30 + 4118523.52) * 0.5  # Y
        self.ship_state[2] = 30 * np.pi / 180.0  # ψ

        # 시뮬레이션 step
        self.dt = 0.01  
        self.con_dt = 0.5  

        self.now_delta = 0.0
        self.now_F = 0.0

        self.nn = 0
        self.tt = 0

        # 입력 지연 설정
        self.delay = 3.0   # 2초 delay
        self.delay_step = int(self.delay / self.con_dt)
        self.input_buffer = np.zeros((self.delay_step, 2))  # [δ_cmd, F_cmd] 히스토리
        self.delayed_input = np.array([0,0])
        self.disturbance_state = np.zeros((6))

        # Timer
        self.timer = self.create_timer(self.dt, self.run)

        # 좌표 변환
        self.utm_proj = pyproj.Proj(proj="utm", zone=52, datum="WGS84")
        self.wgs84_proj = pyproj.Proj(proj="latlong", datum="WGS84")

    def control_callback(self, msg):
        """actuator_outputs → [steer_pwm, thrust_pwm, bow, 0]"""
        steer_pwm = msg.data[0]
        thrust_pwm = msg.data[1]

        # PWM → δ, F 변환
        delta_cmd = self.convert_pwm_to_steering(steer_pwm)
        F_cmd = self.convert_pwm_to_thrust(thrust_pwm)
        self.now_F = F_cmd
        self.now_delta = delta_cmd
        # 새로운 입력을 버퍼에 추가
        self.input_buffer = np.vstack([self.input_buffer[1:], [delta_cmd, F_cmd]])
        self.delayed_input = self.input_buffer[0]
        # print(steer_pwm, thrust_pwm)
        # print(delta_cmd, F_cmd)
        # print(self.input_buffer[-1])
        # print("Time : ", self.tt*self.dt)

    def convert_pwm_to_steering(self, pwm):
        # print(pwm)
        if pwm >= 2000.0: return 300.0
        elif pwm >= 1550.0 and pwm < 2000.0: return (pwm - 1550.0) / 1.6667
        elif pwm <= 1450.0 and pwm > 1000.0: return (pwm - 1450.0) / 1.6667
        elif pwm <= 1000.0: return -300.0
        else: return 0.0

    # def convert_pwm_to_thrust(self, pwm):
    #     if pwm < 1450.0:
    #         return (pwm - 1450.0) / 3.9
    #     elif pwm >= 1550.0:
    #         return (pwm - 1550.0) / 3.9
    #     else:
    #         return 0.0

    def convert_pwm_to_thrust(self, pwm):
        """Inverse of convert_thrust_to_pwm: PWM → rpm_thrust (with deadzone)."""

        # Deadzone center
        if 1450.0 < pwm < 1550.0:
            return 0.0

        # Negative thrust region
        elif pwm <= 1450.0:
            thrust = (pwm - 1450.0) / 3.9   # linear thrust
            thrust_2 = -(thrust**2)         # 복원된 thrust_2 (음수 영역)
            rpm_thrust = 0.0#thrust_2 - (22.0**2)  # deadzone 보정

        # Positive thrust region
        elif pwm >= 1550.0+50:
            thrust = (pwm - 1550.0) / 3.9   # linear thrust
            thrust_2 = (thrust**2)          # 복원된 thrust_2 (양수 영역)
            rpm_thrust = thrust_2 - (22.0**2)  # deadzone 보정

        else:
            rpm_thrust = 0.0

        return 0.01*rpm_thrust

    def utm_to_latlon(self, utm_x, utm_y):
        """Convert UTM coordinates to latitude and longitude."""
        lon, lat = pyproj.transform(self.utm_proj, self.wgs84_proj, utm_x, utm_y)
        return lat, lon

    def wave_disturbance(self, disturbance_state, wave_direction, wind_speed, omega, lamda, Kw, sigmaF1, sigmaF2, dt):
        omega_e = np.abs(omega - (omega * omega / 9.81) * wind_speed * np.cos(wave_direction))
        x1 = disturbance_state[0]
        x2 = disturbance_state[1]

        omegaF1 = np.random.normal(0.0, sigmaF1)
        omegaF2 = np.random.normal(0.0, sigmaF2)

        xdot = np.array([x2, -omega_e * omega_e * x1 - 2 * lamda * omega_e * x2 + Kw * omegaF1, omegaF2])
        disturbance_state = xdot * dt + disturbance_state

        disturbance_force = disturbance_state[1] + disturbance_state[2]
        return disturbance_state, disturbance_force

    def ship_dynamics(self, state, control, dt):
        """컨트롤러 모델과 맞춘 선박 dynamics"""
        # Parameters
        M, I = 1.0, 1.0
        # Xu_dot, Xu, Xuu = 22.43, 1.671, 0.481
        # Yv, Yr = 0.042, 0.0
        # Nr, Nrr = 0.1875, 2.647
        # bu, b2, b3 = 2.03/500.0, 0.01/500.0, 1.4
        # Xu = 15.93
        # Xuu = 0.018
        # Yv = 0.0005
        # Yvv = 0.674
        # Yr = 1.09
        # Nr = 13.67
        # Nrr = 24.577
        # Nv = 1.227
        # alpha1 = 4.467
        # alpha2 = 1.95/500.0
        # alpha3 = 0.049
        # alpha4 = 0.447

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

        # States
        x, y, psi, u, v, r = state
        delta, F_cmd = control
        
        delta = self.now_delta
        # print(delta)
        # F_cmd = self.now_F
        
        # Deadzone + thrust nonlinear
        s, kappa = 25, 8
        a1 = a2 = 2.2**2
        T = ((1/(1+np.exp(s*F_cmd)))*(F_cmd + np.tanh(kappa*F_cmd)*a1) +
             (1/(1+np.exp(-s*F_cmd)))*(F_cmd + np.tanh(kappa*F_cmd)*a2))

        # Dynamics
        # u_dot = (-Xu*u - Xuu*np.sqrt(u*u+eps)*u + T*np.cos(bu*delta)) / (M+Xu_dot) #- 0.08
        # v_dot = (-Yv*v - Yr*r + T*np.sin(b2*delta)) #- 0.01
        # r_dot = (-Nr*r - Nrr*np.sqrt(r*r+eps)*r - b3*T*np.sin(b2*delta))

        ################################################ Environmental disturbance ################################################
        wind_direction = 45*3.141592/180
        wind_speed = 2.0

        self.disturbance_state[:3], XY_wave_force = self.wave_disturbance(
            self.disturbance_state[:3], wind_direction, wind_speed, 0.8, 0.1, 0.64, 6, 2, dt)
        self.disturbance_state[3:6], N_wave_force = self.wave_disturbance(
            self.disturbance_state[3:6], wind_direction, wind_speed, 0.8, 0.1, 1.0, 1, 0.1, dt)

        X_wave_force = XY_wave_force * np.cos(wind_direction - psi)
        Y_wave_force = XY_wave_force * np.sin(wind_direction - psi)
        
        U_wave_force = X_wave_force * np.cos(psi) + Y_wave_force * np.sin(psi)
        V_wave_force = -X_wave_force * np.sin(psi) + Y_wave_force * np.cos(psi)

        # Wind disturbance
        L = 8.0
        H = 2.0
        W = 2.6
        Afw = H*W
        Alw = L*H
        LOA = L
        lau = 1.2
        CD_lAF = 0.55
        delta_ = 0.6
        CDl = CD_lAF * Afw / Alw
        CDt = 0.8

        u_rel_wind = u - wind_speed * np.cos(wind_direction - psi)
        v_rel_wind = v - wind_speed * np.sin(wind_direction - psi)
        gamma = -np.arctan2(v_rel_wind, u_rel_wind)

        Cx = CD_lAF * np.cos(gamma) / (1 - delta_ * 0.5 * (1 - CDl / CDt) * (np.sin(2 * gamma))**2)
        Cy = CDt * np.sin(gamma) / (1 - delta_ * 0.5 * (1 - CDl / CDt) * (np.sin(2 * gamma))**2)
        Cn = -0.18 * (gamma - np.pi * 0.5) * Cy

        X_wind_force = 0.5 * lau * (u_rel_wind**2 + v_rel_wind**2) * Cx * Afw
        Y_wind_force = 0.5 * lau * (u_rel_wind**2 + v_rel_wind**2) * Cy * Alw
        N_wind_force = 0.5 * lau * (u_rel_wind**2 + v_rel_wind**2) * Cn * Alw * LOA

        U_wind_force = X_wind_force * np.cos(psi) + Y_wind_force * np.sin(psi)
        V_wind_force = -X_wind_force * np.sin(psi) + Y_wind_force * np.cos(psi)

        u_dis = 0.005*U_wind_force + 2.0*U_wave_force
        v_dis = 0.005*V_wind_force + 2.0*V_wave_force
        n_dis = 0.005*N_wind_force

        ################################################ Environmental disturbance ################################################
        u_dot = (- Xu*u - Xuu * np.sqrt(u * u + eps) * u + alpha1*T*np.cos(alpha2*delta) + alpha1*u_dis)  
        v_dot = (-Yv*v - Yr*r - Yvv * np.sqrt(v* v + eps) * v + alpha3*T*np.sin(alpha2*delta) + alpha3*v_dis) 
        r_dot = (- Nr*r - Nv*v - Nrr * np.sqrt(r * r + eps) * r - alpha4*T*np.sin(alpha2*delta) + alpha3*n_dis)
        # print(u, v, r, F_cmd)

        # Kinematics
        x_dot = u*np.cos(psi) - v*np.sin(psi)
        y_dot = u*np.sin(psi) + v*np.cos(psi)
        psi_dot = r

        # Forward Euler integration
        next_state = np.zeros_like(state)
        next_state[0] = x + dt*x_dot
        next_state[1] = y + dt*y_dot
        next_state[2] = psi + dt*psi_dot
        next_state[3] = u + dt*u_dot
        next_state[4] = v + dt*v_dot
        next_state[5] = r + dt*r_dot


        # wrap heading
        if next_state[2] > np.pi: next_state[2] -= 2*np.pi
        if next_state[2] < -np.pi: next_state[2] += 2*np.pi

        return next_state

    def run(self):
        """메인 시뮬레이션 루프"""
  
        self.ship_state = self.ship_dynamics(self.ship_state, self.delayed_input, self.dt)
        # print("buffer : ",self.input_buffer)
        # print("delayed_input : ",self.delayed_input)
        # Convert UTM to Lat/Lon
        lat, lon = self.utm_to_latlon(self.ship_state[0], self.ship_state[1])

        # Prepare the state message to publish
        state_msg = Float64MultiArray()
        # Now send lat, lon along with the ship's state (psi, u, v, r)
        state_msg.data = np.concatenate((self.ship_state,self.ship_state)).tolist()
        # state_msg.data = np.concatenate(([lat, lon], self.ship_state[2:],[0,0,0,0,0,0,0,0,0,0,0])).tolist()

        if self.nn >= 10:
            self.state_pub.publish(state_msg)
            self.nn = 0

        self.nn += 1
        self.tt += 1

        

        self.publish_imu()



    def publish_imu(self):
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = "base_link"
        imu_msg.angular_velocity.z = self.ship_state[5]
        imu_msg.linear_acceleration.x = self.ship_state[3]
        imu_msg.linear_acceleration.y = self.ship_state[4]
        self.imu_pub.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    simulator = ShipSimulator()
    rclpy.spin(simulator)
    simulator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
