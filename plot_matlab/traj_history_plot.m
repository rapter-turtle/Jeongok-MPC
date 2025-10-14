clc; clear; close all;

% 데이터를 불러옵니다.
data = readtable('250704_final.csv'); 

x = data.x;
y = data.y;
psi = data.p;
u = data.u;
v = data.v;
r = data.r;
u1 = data.throttle;
u2 = data.steering;

dt = 0.1; 

start_idx = 1; % 시작 인덱스
end_idx = 36000;

% 시간 영역에 해당하는 x, y 값 추출
x_selected = x(start_idx:end_idx);
y_selected = y(start_idx:end_idx);
psi_selected = wrapToPi(psi(start_idx:end_idx));  % 실제 psi 값
u_selected = u(start_idx:end_idx);
v_selected = v(start_idx:end_idx);
r_selected = r(start_idx:end_idx);
u1_selected = u1(start_idx:end_idx);
u2_selected = u2(start_idx:end_idx);

pwm1_selected = arrayfun(@(x) convertThrustToPwm(x), u1_selected)/10;
pwm2_selected = arrayfun(@(x) convertSteeringToPwm(x), u2_selected)/500;

% x, y 값의 중앙값 계산 (중앙값 기준으로 그래프를 이동)
x_center = mean(x_selected);
y_center = mean(y_selected);

% x, y 값을 중앙값을 기준으로 이동
x_selected = x_selected - x_center;
y_selected = y_selected - y_center;

% x, y 값의 최소값과 최대값 구하기
x_min = min(x_selected);
x_max = max(x_selected);
y_min = min(y_selected);
y_max = max(y_selected);

% x, y 축 범위 설정 (각각 30만큼 더하고 빼기)
x_range = [-300, 400];
y_range = [-250, 200];

% 색상 설정: 구간별 색상 배열 생성
colors = lines(4); % 4개의 색상
legend_labels = {'Section 1', 'Section 2', 'Section 3', 'Section 4'};

% 그래프 그리기
figure;
hold on;

i1 = 5500;
i2 = 8500;
i3 = 15000;
i4 = 25000;
i5 = 31500;
i6 = 34300;
i7 = 36000;
% 각 구간을 직접 지정
% Section 1: 인덱스 5000 ~ 5200
idx1 = 5000:i1;
plot(x_selected(idx1), y_selected(idx1), 'LineWidth', 1.0, 'Color', 'k');

idx3 = i2+1:i3;
plot(x_selected(idx3), y_selected(idx3), 'LineWidth', 1.0, 'Color', 'k');

idx6 = i5+1:i6;
plot(x_selected(idx6), y_selected(idx6), 'LineWidth', 1.0, 'Color', 'k');

idx2 = i1+1:i2;
h2 = plot(x_selected(idx2), y_selected(idx2), 'LineWidth', 2.0, 'Color', colors(1,:), 'DisplayName', 'Constant thrust, Acc-Dcc');

idx4 = i3+1:i4;
h4 = plot(x_selected(idx4), y_selected(idx4), 'LineWidth', 2.0, 'Color', colors(2,:), 'DisplayName', 'Zigzag');

idx5 = i4+1:i5;
h5 = plot(x_selected(idx5), y_selected(idx5), 'LineWidth', 2.0, 'Color', colors(3,:), 'DisplayName', 'Circling');


idx7 = i6+1:i7;
plot(x_selected(idx7), y_selected(idx7), 'LineWidth', 2.0, 'Color', colors(3,:), 'DisplayName', 'Circling');



xlabel('X [m]');
ylabel('Y [m]');
grid on;
axis([x_range, y_range]); % x, y 축 범위 설정
axis equal;

xticks(-300:100:400); % x축 간격 설정
yticks(-250:100:200); % y축 간격 설정

xlim(x_range);
ylim(y_range);

legend([h2, h4, h5], {'Constant thrust, Acc-Dcc', 'Zigzag', 'Circling'});

set(gca, 'Box', 'on', 'LineWidth', 0.5, 'EdgeColor', 'k');
