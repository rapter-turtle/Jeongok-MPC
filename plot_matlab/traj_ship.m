clc; clear; close all;

% 데이터를 불러옵니다.
% data = readtable('250704_final.csv'); 
% data = readtable('030623_final.csv'); 
% data = readtable('0731_8.csv'); 
% data = readtable('0731_8_dob.csv'); 
% data = readtable('siwbow.csv');
% data = readtable('full_nom.csv'); 
% data = readtable('wpt1.csv');  
% data = readtable('wpt2.csv');  
% data = readtable('wpt3.csv');
data = readtable('w_dob2.csv');  
% data = readtable('small_nom.csv');  
% data = readtable('w_dob2.csv');  

% data = readtable('pinn_zigzag4.csv');
% data = readtable('pinn_const_thrust.csv');  

x = data.x;
y = data.y;
psi = data.p;
u = data.u;
v = data.v;
r = data.r;
u1 = data.throttle;
u2 = data.steering;

% x = data.Var2;
% y = data.Var3;
% psi = data.Var4;
% u = data.Var5;
% v = data.Var6;
% r = data.Var7;
% u1 = data.Var15;
% u2 = data.Var14;


% refx = data.ref_x;
% refy = data.ref_y;


% 최적화된 파라미터 값
b1 = 1.0;
% bu = 2.35;
% xdot = -0.98417;

% xdot: 22.4301
% tau: 0.32256
% bu: 2.03

bu = 2.03;%1.7417;
xdot = 22.43;
% xdot = 30.2;
tau = 1/0.322;%1.2;
Xu = 1.671;
Xuu = 0.481;
b1u = 1.0;

%PINN
% bu = 2.91;%1.7417;
% xdot = 0.0;
% tau = 1;%1.2;
% Xu = 0.194;
% Xuu = 0.0232;
% b1u = 1.0;

% Xu = 2.635374;
% Xuu = 0.748663;

% Yv= 0.10739;
% Yn= 0.0;%1.7935e-08;
% Nr= 0.3478;
% Nv= 0.0;%7.2436e-09;
% Nrr= 0.29961;
% Yvv= 0.0;%4.1296e-09;
% b3= 0.57483;
% b2= 0.044797;

% Yv= 0.17;%0.13616;
% Yn= 4.8161e-08;
% Nr= 1.8422;
% Nv= 0.076233;
% Nrr= 2.9771e-07;
% Yvv= 1.5407e-09;
% b3= 1.53;
% b2= 0.052779;

% % 8자
Yv= 0.042;
Yn= 0.0;%1.7935e-08;
Nr= 0.1875;
Nv= 0.0;%7.2436e-09;
Nrr= 2.647;
Yvv= 0.0;%4.1296e-09;
b3= 1.4;
b2= 0.0102;

% Slow speed
% Yv = 0.0038746;
% Yn = 0.73973;
% Nr = 0.053878;
% Nv = 0.033494;
% Nrr = 1.404;
% Yvv = 0.066574;
% b3 = 1.4684;
% b2 = 0.0088565;

% PINN
% Yv = 2.216;
% Yn = 5.22;
% Nr = 1.867;
% Nv = 0.518;
% Nrr = 0.189;
% Yvv = 0.133;
% b3 = 0.0205;
% b2 = 2.91;


dt = 0.1; 

% start_idx = 7000; % 시작 인덱스
% end_idx = 8400;
start_idx = 10; % 시작 인덱스
end_idx = 30000;


% 시간 배열 생성 (dt = 0.1)
t = (start_idx:end_idx) * dt;% - start_idx*dt;  % 시간 배열 (0.1초 간격)

% 시간 영역에 해당하는 x, y 값 추출
x_selected = x(start_idx:end_idx) - x(start_idx);
y_selected = y(start_idx:end_idx) - y(start_idx);
psi_selected = wrapToPi(psi(start_idx:end_idx));  % 실제 psi 값
u_selected = u(start_idx:end_idx);
v_selected = v(start_idx:end_idx);
r_selected = r(start_idx:end_idx);
u1_selected = u1(start_idx:end_idx);
u2_selected = u2(start_idx:end_idx);

pwm1_selected = arrayfun(@(x) convertThrustToPwm(x), u1_selected)/10;
pwm2_selected = arrayfun(@(x) convertSteeringToPwm(x), u2_selected)/500;

% 예측 값 초기화
psi_predicted = zeros(1, length(x_selected));
u_predicted = zeros(1, length(x_selected));
v_predicted = zeros(1, length(x_selected));
r_predicted = zeros(1, length(x_selected));
x_predicted = zeros(1, length(x_selected));
y_predicted = zeros(1, length(x_selected));
u_eff_predicted = zeros(1, length(x_selected));

u_predicted(1) = u_selected(1); % 초기값 설정
v_predicted(1) = v_selected(1);
r_predicted(1) = r_selected(1);
x_predicted(1) = x_selected(1);
y_predicted(1) = y_selected(1);
psi_predicted(1) = psi_selected(1);
a = 1.0;
u_eff_predicted(1) = b1*a*pwm1_selected(1)*pwm1_selected(1);

for i = 2:length(t)
    % 시스템 동역학을 기반으로 다음 상태를 예측 (최적화된 계수 사용)


    % v_predicted(i) = (-Yv_opt*v_predicted(i-1) - Yvv_opt*abs(v_predicted(i-1))*v_predicted(i-1) - Yn_opt*r_predicted(i-1) + u_selected(i-1)*v_predicted(i-1) + b1*pwm1_selected(i)*pwm1_selected(i-1)*sin(b2_opt*pwm2_selected(i-1)))*dt+ v_predicted(i-1);
    % r_predicted(i) = (-Nr_opt*r_predicted(i-1) - Nrr_opt*abs(r_predicted(i-1))*r_predicted(i-1) - Nv_opt*v_predicted(i-1) + u_selected(i-1)*v_predicted(i-1) - u_selected(i-1)*r_predicted(i-1) - b1*b3_opt*pwm1_selected(i-1)*pwm1_selected(i-1)*sin(b2_opt*pwm2_selected(i-1)))*dt + r_predicted(i-1);

    v_predicted(i) = (-Yv*v_predicted(i-1) - Yvv*abs(v_predicted(i-1))*v_predicted(i-1) - Yn*r_predicted(i-1) + b1*pwm1_selected(i)*pwm1_selected(i-1)*sin(b2*pwm2_selected(i-1)))*dt+ v_predicted(i-1);
    r_predicted(i) = (-Nr*r_predicted(i-1) - Nrr*abs(r_predicted(i-1))*r_predicted(i-1) - Nv*v_predicted(i-1) - b1*b3*pwm1_selected(i-1)*pwm1_selected(i-1)*sin(b2*pwm2_selected(i-1)))*dt + r_predicted(i-1);


end

for i = 2:length(x_selected)
    % 시스템 동역학을 기반으로 다음 상태를 예측 (최적화된 계수 사용)
    if pwm1_selected(i-1) >= 0
        a = 1.0;
    else
        a = -1.0;
    end
    du_eff = (b1*a*pwm1_selected(i-1)*pwm1_selected(i-1)*cos(bu*pwm2_selected(i-1)) - u_eff_predicted(i-1))/tau;
    u_eff_predicted(i) = u_eff_predicted(i-1) + du_eff*0.1;

    du = (-Xu*u_predicted(i-1) - Xuu*abs(u_predicted(i-1))*u_predicted(i-1) + b1u*u_eff_predicted(i))/(1.0 + xdot);
    u_predicted(i) = du*dt + u_predicted(i-1);    
    % % v_predicted(i) = (-Yv*v_predicted(i-1) - Yvv*abs(v_predicted(i-1))*v_predicted(i-1) - Yn*r_predicted(i-1) + b1*a*u_eff_predicted(i)*sin(b2*pwm2_selected(i-1)))*dt + v_predicted(i-1);
    % % r_predicted(i) = (-Nr*r_predicted(i-1) - Nrr*abs(r_predicted(i-1))*r_predicted(i-1) - Nv*v_predicted(i-1) - b3*b1*a*u_eff_predicted(i)*sin(b2*pwm2_selected(i-1)))*dt + r_predicted(i-1);
    % v_predicted(i) = (-Yv*v_predicted(i-1) - Yvv*abs(v_predicted(i-1))*v_predicted(i-1) - Yn*r_predicted(i-1) + b1*pwm1_selected(i)*pwm1_selected(i-1)*sin(b2*pwm2_selected(i-1)))*dt+ v_predicted(i-1);
    % r_predicted(i) = (-Nr*r_predicted(i-1) - Nrr*abs(r_predicted(i-1))*r_predicted(i-1) - Nv*v_predicted(i-1) - b1*b3*pwm1_selected(i-1)*pwm1_selected(i-1)*sin(b2*pwm2_selected(i-1)))*dt + r_predicted(i-1);
   
    % psi_predicted(i) = wrapToPi(psi_predicted(i-1) + r_predicted(i)*dt);
    % 
    % % x, y 값 예측
    % x_predicted(i) = x_predicted(i-1) + (u_predicted(i)*cos(psi_predicted(i)) - v_predicted(i)*sin(psi_predicted(i)))*dt;
    % y_predicted(i) = y_predicted(i-1) + (u_predicted(i)*sin(psi_predicted(i)) + v_predicted(i)*cos(psi_predicted(i)))*dt; 
end

% x, y 값의 최소값과 최대값 구하기
x_min = min([x_selected']);
x_max = max([x_selected']);
y_min = min([y_selected']);
y_max = max([y_selected']);

% x, y 축 범위 설정 (각각 5만큼 더하고 빼기)
x_range = [x_min - 30, x_max + 30];
y_range = [y_min - 30, y_max + 30];

figure;
% plot(x_selected, y_selected, 'b', 'LineWidth', 0.5, 'DisplayName', 'traj actual'); hold on;
% xlabel('X [m]');
% ylabel('Y [m]');
% % legend;
% % title('Comparison of Actual and Predicted Trajectory');
% grid on;
% axis([x_range, y_range]); % x, y 축 범위 설정
% axis equal;


% % 왼쪽: 궤적 비교 (전체 영역을 차지)
% subplot(6,2,[1, 3, 5, 7, 9]); % 1행 2열 중 첫 번째 subplot
% plot(x_selected, y_selected, 'b', 'LineWidth', 0.5, 'DisplayName', 'traj actual'); hold on;
% % plot(x_predicted, y_predicted, 'r', 'LineWidth', 0.5, 'DisplayName', 'traj pred');
% xlabel('x');
% ylabel('y');
% legend;
% title('Comparison of Actual and Predicted Trajectory');
% grid on;
% axis([x_range, y_range]); % x, y 축 범위 설정
% axis equal;
% 
% % 오른쪽: 상태 변수 비교 (2행 3열로 나누어 표시)
% subplot(6,2,2); % 첫 번째 subplot (psi 비교)
% plot(t, psi_selected, 'b', 'DisplayName', 'psi actual'); hold on;
% plot(t, psi_predicted, 'r--', 'DisplayName', 'psi predicted');
% xlabel('Time [s]');
% ylabel('psi');
% legend;
% title('Comparison of psi actual and psi predicted');
% 
% subplot(6,2,4); % 두 번째 subplot (u 비교)
% plot(t, u_selected, 'b', 'DisplayName', 'u actual'); hold on;
% plot(t, u_predicted, 'r--', 'DisplayName', 'u predicted');
% xlabel('Time [s]');
% ylabel('u [m/s]');
% % ylim([0,13.0]);
% legend;
% % title('Comparison of u actual and u predicted');
% 
% subplot(6,2,6); % 세 번째 subplot (v 비교)
% plot(t, v_selected, 'b', 'DisplayName', 'v actual'); hold on;
% plot(t, v_predicted, 'r--', 'DisplayName', 'v predicted');
% xlabel('Time [s]');
% ylabel('v [m/s]');
% ylim([-1,1.5]);
% legend;
% % title('Comparison of v actual and v predicted');
% 
% subplot(6,2,8); % 네 번째 subplot (r 비교)
% plot(t, r_selected, 'b', 'DisplayName', 'r actual'); hold on;
% plot(t, r_predicted, 'r--', 'DisplayName', 'r predicted');
% xlabel('Time [s]');
% ylabel('r [rad/s]');
% ylim([-0.25,0.25]);
% legend;
% % title('Comparison of r actual and r predicted');
% 
% subplot(6,2,10); % 다섯 번째 subplot (u1 비교)
% plot(t, pwm1_selected, 'b', 'DisplayName', 'u1 actual'); hold on;
% xlabel('Time [s]');
% ylabel('u1');
% legend;
% title('Comparison of u1 actual and u1 predicted');
% 
% subplot(6,2,12); % 여섯 번째 subplot (u2 비교)
% plot(t, pwm2_selected, 'b', 'DisplayName', 'u2 actual'); hold on;
% xlabel('Time [s]');
% ylabel('u2');
% legend;
% title('Comparison of u2');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
subplot(6,1,1); % 두 번째 subplot (u 비교)
plot(t, u_selected, 'b', 'DisplayName', 'u actual', 'LineWidth', 2); hold on;
plot(t, u_predicted, 'r--', 'DisplayName', 'u predicted', 'LineWidth', 2);
xlabel('Time [s]', 'FontSize', 10);
ylabel('u [m/s]', 'FontSize', 10);
% ylim([2.2, 3.2]);
yticks(2.2:0.5:3.2);
legend('FontSize', 12);
set(gca, 'FontSize', 13);

subplot(6,1,2); % 세 번째 subplot (v 비교)
plot(t, v_selected, 'b', 'DisplayName', 'v actual', 'LineWidth', 2); hold on;
plot(t, v_predicted, 'r--', 'DisplayName', 'v predicted', 'LineWidth', 2);
xlabel('Time [s]', 'FontSize', 10);
ylabel('v [m/s]', 'FontSize', 10);
% ylim([-1.5, 1.5]);
legend('FontSize', 12);
set(gca, 'FontSize', 13);

subplot(6,1,3); % 네 번째 subplot (r 비교)
plot(t, r_selected, 'b', 'DisplayName', 'r actual', 'LineWidth', 2); hold on;
plot(t, r_predicted, 'r--', 'DisplayName', 'r predicted', 'LineWidth', 2);
xlabel('Time [s]', 'FontSize', 10);
ylabel('r [rad/s]', 'FontSize', 10);
% ylim([-0.3, 0.3]);
legend('FontSize', 12);
set(gca, 'FontSize', 13);

subplot(6,1,4); % 첫 번째 subplot (psi 비교)
plot(t, psi_selected*180/3.141592, 'b', 'DisplayName', 'psi actual'); hold on;
% plot(t, psi_predicted, 'r--', 'DisplayName', 'psi predicted');
xlabel('Time [s]');
ylabel('psi');
legend;

subplot(6,1,5); % 다섯 번째 subplot (u1 비교)
plot(t, pwm1_selected, 'b', 'DisplayName', 'u1 actual'); hold on;
xlabel('Time [s]', 'FontSize', 10);
ylabel('u1', 'FontSize', 10);
legend('FontSize', 12);
set(gca, 'FontSize', 13);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure;
% subplot(2,1,1); % 두 번째 subplot (u 비교)
% plot(t, u_selected, 'b', 'DisplayName', 'u actual', 'LineWidth', 2); hold on;
% % plot(t, u_predicted, 'r--', 'DisplayName', 'u predicted', 'LineWidth', 2);
% xlabel('Time [s]', 'FontSize', 10);
% ylabel('u [m/s]', 'FontSize', 10);
% % ylim([2.2, 3.2]);
% % yticks(2.2:0.5:3.2);
% legend('FontSize', 12);
% set(gca, 'FontSize', 13);
% 
% subplot(2,1,2); % 다섯 번째 subplot (u1 비교)
% plot(t, pwm1_selected, 'b', 'DisplayName', 'u1 actual'); hold on;
% xlabel('Time [s]', 'FontSize', 10);
% ylabel('u1', 'FontSize', 10);
% legend('FontSize', 12);
% set(gca, 'FontSize', 13);

