clc; clear; close all;

% 데이터를 불러옵니다.
% data = readtable('original_code\J1.csv'); % -- mid
% data = readtable('original_code\J2.csv'); % -- mid
% data = readtable('original_code\rosbag2_2025_09_29_22_42_54.csv'); % -- mid
% data = readtable('original_code\rosbag2_2025_09_30-17_27_38.csv'); % -- big
% data = readtable('original_code\rosbag2_2025_09_30-19_32_28.csv'); % -- small
% data = readtable('original_code\rosbag2_2025_09_30-19_59_04.csv'); % -- small

data = readtable('mid_mpc.csv');
% data = readtable('mid_dob_mpc.csv');
% data = readtable('mid_dt_mpc_1.csv');
% data = readtable('mid_dob_dt_mpc_2.csv');

% data = readtable('big_mpc.csv');
% data = readtable('big_dob_mpc.csv');
% data = readtable('big_dt_mpc_1.csv');
% data = readtable('big_dob_dt_mpc_2.csv');

% data = readtable('small_mpc_1.csv');
% data = readtable('small_dob_mpc_1.csv');
% data = readtable('small_dt_mpc_2.csv');
% data = readtable('small_dob_dt_mpc_1.csv');

start_idx = 1; 
end_idx = 5100;

x = data.x;
y = data.y;
psi = data.p;
u = data.u;
v = data.v;
r = data.r;
u1 = data.throttle;
u2 = data.steering;
refx = data.ref_x;
refy = data.ref_y;
d_u = data.du;
d_v = data.dv;
d_r = data.dr;

dt = 0.1; 

%%%%%%%%%%%%%%%%%%% mid %%%%%%%%%%%%%%%%%%%
% J1_1  % DT mpc
% start_idx = 7330; 
% end_idx = 15000;

% J2_1 %Dt mpc
% start_idx = 1; 
% end_idx = 5250;

% % J2_2 %DOB dt mpc
% start_idx = 5270; 
% end_idx = 10000;

% rosbag2_2025_09_29_22_42_54  1  %% mpc
% start_idx = 25280; 
% end_idx = 30000;

% rosbag2_2025_09_29_22_42_54  2 %% dt mpc
% start_idx = 32924; 
% end_idx = 38000;

% rosbag2_2025_09_29_22_42_54  3 %% dob dt mpc
% start_idx = 38890; 
% end_idx = 46000;

% rosbag2_2025_09_29_22_42_54  4 $$ dob mpc
% start_idx = 47090; 
% end_idx = 55000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% big %%%%%%%%%%%%%%%%%%%

% rosbag2_2025_09_30-17_27_38  1 %% mpc
% start_idx = 14070; 
% end_idx = 28450;

% rosbag2_2025_09_30-17_27_38  2 %% dob mpc
% start_idx = 44200; 
% end_idx = 53500;

% % rosbag2_2025_09_30-17_27_38  3 %% dmpc
% start_idx = 58742; 
% end_idx = 68300;

% % rosbag2_2025_09_30-17_27_38  4 %% dob dt mpc
% start_idx = 68440; 
% end_idx = 78100;

% % rosbag2_2025_09_30-17_27_38  5 %% dt mpc
% start_idx = 80336; 
% end_idx = 89900;

% % rosbag2_2025_09_30-17_27_38  6  %% dob dt mpc
% start_idx = 94000; 
% end_idx = 104900;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% small %%%%%%%%%%%%%%%%%%%

% rosbag2_2025_09_30-19_32_28 1 %% mpc
% start_idx = 2550; 
% end_idx = 6000;

% rosbag2_2025_09_30-19_32_28 2 %% dt mpc
% start_idx = 6510; 
% end_idx = 12300;

% rosbag2_2025_09_30-19_32_28 3 %% mpc
% start_idx = 13350; 
% end_idx = 18270;

% rosbag2_2025_09_30-19_59_04 1 %% dob dt mpc 
% start_idx = 50; 
% end_idx = 6100;

% rosbag2_2025_09_30-19_59_04 2 %% dob mpc
% start_idx = 11700; 
% end_idx = 15600;

% rosbag2_2025_09_30-19_59_04 3 %% dt mpc
% start_idx = 23980; 
% end_idx = 28800;

% rosbag2_2025_09_30-19_59_04 4 %% dob mpc
% start_idx = 29100; 
% end_idx = 34200;

% rosbag2_2025_09_30-19_59_04 5 %% dt mpc
% start_idx = 34350; 
% end_idx = 39350;

% rosbag2_2025_09_30-19_59_04 6 %% dob dt mpc
% start_idx = 39370; 
% end_idx = 48500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% 시간 배열 생성 (dt = 0.1)
t = (start_idx:end_idx) * dt;  % 시간 배열 (0.1초 간격)

% 시간 영역에 해당하는 x, y 값 추출
x_selected = x(start_idx:end_idx) - x(start_idx);
y_selected = y(start_idx:end_idx) - y(start_idx);
psi_selected = wrapToPi(psi(start_idx:end_idx));  % 실제 psi 값
u_selected = u(start_idx:end_idx);
v_selected = v(start_idx:end_idx);
r_selected = r(start_idx:end_idx);
u1_selected = u1(start_idx:end_idx);
u2_selected = u2(start_idx:end_idx);
refx_selected = refx(start_idx:end_idx) - x(start_idx);
refy_selected = refy(start_idx:end_idx) - y(start_idx);
du_selected = d_u(start_idx:end_idx);
dv_selected = d_v(start_idx:end_idx);
dr_selected = d_r(start_idx:end_idx);

pwm1_selected = arrayfun(@(x) convertThrustToPwm(x), u1_selected)/10;
pwm2_selected = arrayfun(@(x) convertSteeringToPwm(x), u2_selected)/500;



% x, y 값의 최소값과 최대값 구하기
x_min = min([refx_selected']);
x_max = max([refx_selected']);
y_min = min([refy_selected']);
y_max = max([refy_selected']);

% x, y 축 범위 설정 (각각 5만큼 더하고 빼기)
x_range = [x_min - 10, x_max + 10];
y_range = [y_min - 10, y_max + 10];

figure;
% % 왼쪽: 궤적 비교 (전체 영역을 차지)
viz.colors  = {[0 0 1], [0.5 0.1 0.5], [0 0.5 0], [1 0 0]};
plot(x_selected, y_selected, 'Color', [0 0 1], 'LineWidth', 1.7, 'DisplayName', 'traj actual'); hold on;
plot(refx_selected, refy_selected, 'k--', 'LineWidth', 0.5, 'DisplayName', 'traj pred');
xlabel('X [m]');
ylabel('Y [m]');
% legend('Location','best','Orientation','horizontal'); 
% title('Comparison of Actual and Predicted Trajectory');
grid on;
axis equal;
axis([x_range, y_range]); % x, y 축 범위 설정


