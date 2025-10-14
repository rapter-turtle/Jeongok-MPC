%% ================= du, dv, dr Plot (single file) =================
clc; clear; close all;

dt = 0.1;
fname = 'mid_dob_dt_mpc_1.csv';
T = readtable(fname);

% ---- 데이터 확인 ----
need = {'du','dv','dr'};
if ~all(ismember(need, T.Properties.VariableNames))
    error('File %s must contain columns: du, dv, dr', fname);
end

% ---- 구간 설정 ----
idx_range = 1000:4000;   % 원하는 구간만
t_sub = (0:length(idx_range)-1) * dt;

du = T.du(idx_range);
dv = T.dv(idx_range);
dr = T.dr(idx_range);

% ---- 플로팅 ----
figure('Color','w'); hold on; grid on;

% 왼쪽 y축: du, dv
yyaxis left
p1 = plot(t_sub, du, '-','Color',[1 0 0], 'LineWidth',2.0, 'DisplayName','du');
p2 = plot(t_sub, dv, '-','Color',[0 0.5 0], 'LineWidth',2.0, 'DisplayName','dv');
ylabel('Estimated disturbance [m/s^2]', 'FontSize',16, 'FontWeight','bold');
set(gca,'YColor',[0 0 0]);  % 왼쪽 y축 색

% 오른쪽 y축: dr
yyaxis right
p3 = plot(t_sub, dr, 'Color',[0 0 1], 'LineWidth',2.0, 'DisplayName','dr');
ylim([-0.4, 0.3]);
ylabel('Estimated angular disturbance [rad/s^2]', 'FontSize',16, 'FontWeight','bold');
set(gca,'YColor',[0 0 0]);  % 오른쪽 y축 색

% ---- 공통 설정 ----
xlabel('Time [s]', 'FontSize',16, 'FontWeight','bold');
set(gca, 'FontSize',14, 'FontName','Arial', 'LineWidth',1.5);
box on; grid on;

% ---- 범례 (가로 방향) ----
lgd = legend([p1 p2 p3], {'du','dv','dr'}, ...
    'Location','southoutside','Orientation','horizontal');
set(lgd,'FontSize',13,'FontName','Arial');


