%% ================= 추가: du 비교 (특정 구간만) =================
fname1 = 'mid_dob_mpc.csv';
fname2 = 'mid_dob_dt_mpc_1.csv';

T1 = readtable(fname1);
T2 = readtable(fname2);

N = min(height(T1), height(T2));
du1 = T1.du(1:N);
du2 = T2.du(1:N);
t_du = (0:N-1)*dt;

% 🔹 사용하고 싶은 구간 설정
idx_range = 1000:4000;   % <-- 원하는 구간만 지정
idx_range_t = 1:3001;

t_sub  = t_du(idx_range_t);
du1_sub = du1(idx_range);
du2_sub = du2(idx_range);


% 플로팅
figure('Color','w'); hold on; grid on;
plot(t_sub, du1_sub, 'Color',[0.5 0.1 0.5], 'LineWidth',2.0, 'DisplayName','DOB-MPC');
plot(t_sub, du2_sub, 'Color',[1 0 0], 'LineWidth',2.0, 'DisplayName','DTC-DOB-MPC');

% ---- 폰트 및 라벨 크기 설정 ----
xlabel('Time [s]', 'FontSize', 16, 'FontWeight','bold');
ylabel('Estimated disturbance [m/s^2]', 'FontSize', 16, 'FontWeight','bold');
set(gca, 'FontSize', 14, 'FontName', 'Arial', 'LineWidth', 1.5);  % 축 눈금, 테두리

% ---- 범례 (가로 방향 + 폰트 크게) ----
lgd = legend('Location','southoutside','Orientation','horizontal');
set(lgd, 'FontSize', 13, 'FontName', 'Arial');

box on; grid on;

