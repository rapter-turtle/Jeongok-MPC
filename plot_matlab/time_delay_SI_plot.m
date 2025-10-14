function plot_identified_model()
    clc; clear; close all;

    %% --- 추정된 파라미터 ---
    params = struct();
    params.Xu     = 0.0845;
    params.Xuu    = 0.0195;
    params.Yv     = 0.048458;
    params.Yr     = 0.151;
    params.Yvv    = 0.0988;
    params.Nr     = 0.6939;
    params.Nv     = 0.0;
    params.Nrr    = 0.0;
    params.alpha1 = 0.0452;
    params.alpha2 = 2.0;
    params.alpha3 = 0.0188;
    params.alpha4 = 0.01932;    

    dt = 0.1;
    delay_steps = round(3.0/dt);   % 3초 지연 (throttle만 적용)

    %% --- 데이터 로드 ---
    % T = readtable('dataset/all/pinn_zigzag3.csv');
    T = readtable('dataset/train/pinn_const_thrust.csv');

    tau_T = arrayfun(@convertThrustToPwm, T.throttle) / 10;
    tau_S = arrayfun(@convertSteeringToPwm, T.steering) / 500;

    u_true = T.u(:)'; 
    v_true = T.v(:)'; 
    r_true = T.r(:)';

    %% --- rollout (추정된 파라미터로) ---
    % xvec = pack_params(params);
    % 
    % init_u = u_true(1);
    % init_v = v_true(1);
    % init_r = r_true(1);
    % 
    % [u_pred,v_pred,r_pred,tau_T_shifted,tau_S_used,steer_effect] = rollout(xvec, ...
    %                                  init_u, init_v, init_r, ...
    %                                  tau_T, tau_S, dt, delay_steps);
    %% --- rollout (추정된 파라미터로) ---
    xvec = pack_params(params);
    
    dt = 0.1;
    delay_steps = round(3.0/dt);
    
    % --- delay 이후 시점부터 시작하도록 설정 ---
    idx0 = delay_steps + 1;
    init_u = u_true(idx0);
    init_v = v_true(idx0);
    init_r = r_true(idx0);
    
    % delay 이후 입력만 사용 (즉, 실제로 제어가 반영되는 구간)
    tau_T_used = tau_T(1:end);
    tau_S_used = tau_S(idx0:end);
    
    [u_pred,v_pred,r_pred,tau_T_shifted,tau_S_used,steer_effect] = rollout( ...
        xvec, init_u, init_v, init_r, tau_T_used, tau_S_used, dt, delay_steps);
    
    % --- 시간 및 실제 데이터 동기화 ---
    N = length(u_pred);
    t = (1:N)*dt;
    
    % u_true = u_true(idx0:idx0+N-1);
    % v_true = v_true(idx0:idx0+N-1);
    % r_true = r_true(idx0:idx0+N-1);
    % tau_T  = tau_T(idx0:idx0+N-1);
    % tau_S  = tau_S(idx0:idx0+N-1);


    u_true = u_true(idx0:N+delay_steps);
    v_true = v_true(idx0:N+delay_steps);
    r_true = r_true(idx0:N+delay_steps);
    tau_T  = tau_T(idx0:N+delay_steps);
    tau_S  = tau_S(idx0:N+delay_steps);

    length(u_true)
    length(u_pred)
    length(t)
    
    %% --- Surge Plot ---
    figure;

    % ---- (1) Surge velocity plot ----
    subplot(2,1,1);
    plot(t, u_true, 'k', 'LineWidth',1.8); hold on;
    plot(t, u_pred, 'r--', 'LineWidth',1.8);
    ylabel('u [m/s]', 'FontSize', 14);
    xlabel('Time [s]', 'FontSize', 14);
    legend({'True','Predicted'}, 'FontSize', 12, ...
           'Location','southoutside', 'Orientation','horizontal');
    grid on; ylim([3.5, 5.0]);
    set(gca, 'FontSize', 13);  % 축 숫자 및 tick 크기

    % ---- (2) Throttle input plot ----
    subplot(2,1,2);
    plot(t, tau_T*10, 'b', 'LineWidth',1.8); hold on;
    plot(t, tau_T_shifted*10, 'r--', 'LineWidth',1.8);
    ylabel('Throttle input [%]', 'FontSize', 14);
    ylim([35,45]);
    xlabel('Time [s]', 'FontSize', 14);
    legend({'Original','Delayed(3s)'}, 'FontSize', 12, ...
           'Location','southoutside', 'Orientation','horizontal');
    grid on;
    set(gca, 'FontSize', 13);


    %% --- Sway/Yaw Plot (입력+상태) ---
    % figure;
    % 
    % % ---- (1) Sway v ----
    % subplot(4,1,1);
    % plot(t, v_true, 'k', 'LineWidth',1.8); hold on;
    % plot(t, v_pred, 'r--', 'LineWidth',1.8);
    % ylabel('v [m/s]', 'FontSize', 14);
    % ylim([-1.5, 1.5]);
    % xlabel('Time [s]', 'FontSize', 14);
    % legend({'True','Predicted'}, 'FontSize', 12, ...
    %        'Location','southoutside', 'Orientation','horizontal');
    % grid on;
    % set(gca, 'FontSize', 13);
    % 
    % % ---- (2) Yaw r ----
    % subplot(4,1,2);
    % plot(t, r_true, 'k', 'LineWidth',1.8); hold on;
    % plot(t, r_pred, 'r--', 'LineWidth',1.8);
    % ylabel('r [rad/s]', 'FontSize', 14);
    % ylim([-0.3, 0.3]);
    % xlabel('Time [s]', 'FontSize', 14);
    % legend({'True','Predicted'}, 'FontSize', 12, ...
    %        'Location','southoutside', 'Orientation','horizontal');
    % grid on;
    % set(gca, 'FontSize', 13);
    % 
    % % ---- (3) Throttle input ----
    % subplot(4,1,3);
    % plot(t, tau_T*10, 'b', 'LineWidth',1.8); hold on;
    % plot(t, tau_T_shifted*10, 'r--', 'LineWidth',1.8);
    % ylabel('Throttle input [%]', 'FontSize', 14);
    % ylim([28, 36]);
    % xlabel('Time [s]', 'FontSize', 14);
    % legend({'Original','Delayed(3s)'}, 'FontSize', 12, ...
    %        'Location','southoutside', 'Orientation','horizontal');
    % grid on;
    % set(gca, 'FontSize', 13);
    % 
    % % ---- (4) Steering input ----
    % subplot(4,1,4);
    % plot(t, tau_S*50*2, 'b', 'LineWidth',1.8);
    % % ylabel('Steering input [\angle]', 'FontSize', 14);
    % ylabel(['Steering input [', char(176), ']'], 'FontSize', 14);
    % 
    % xlabel('Time [s]', 'FontSize', 14);
    % % legend({'Original'}, 'FontSize', 12, ...
    %        % 'Location','southoutside', 'Orientation','horizontal');
    % grid on;
    % set(gca, 'FontSize', 13);

end

%% ===== rollout 함수 =====
function [u_traj,v_traj,r_traj,tau_T_shifted,tau_S_used,steer_effect] = rollout(x,u0,v0,r0,tau_T_seq,tau_S_seq,dt,delay_steps)
    tau_T_seq = tau_T_seq(:)';  
    tau_S_seq = tau_S_seq(:)';  
    N = length(tau_T_seq);

    % unpack parameters
    Xu   = exp(x(1)); 
    Xuu  = exp(x(2));
    alpha1 = exp(x(3));
    alpha2 = 3/(1+exp(-x(4)));   % sigmoid
    Yv  = exp(x(5));
    Yr  = exp(x(6));
    Yvv = exp(x(7));
    Nr  = exp(x(8));
    Nv  = exp(x(9));
    Nrr = exp(x(10));
    alpha3 = exp(x(11));
    alpha4 = exp(x(12));

    % --- throttle에는 delay 적용, steering은 즉시 반영 ---
    % if N <= delay_steps
    %     tau_T_shifted = zeros(1,N);     % throttle은 지연된 입력
    % else
    %     tau_T_shifted = [zeros(1,delay_steps), tau_T_seq(1:N-delay_steps)];
    % end
    tau_T_shifted = tau_T_seq(1:N - delay_steps);
    tau_S_used = tau_S_seq;             % steering은 원본 그대로 사용

    % init states: [u,v,r]
    state = [u0,v0,r0];

    u_traj = zeros(1,N - delay_steps);
    v_traj = zeros(1,N - delay_steps);
    r_traj = zeros(1,N - delay_steps);
    steer_effect = zeros(1,N);

    u_traj(1) = u0; v_traj(1) = v0; r_traj(1) = r0;

    for k = 1:N-delay_steps-1
        tau_T_delayed = tau_T_shifted(k);
        tau_S_now     = tau_S_used(k);

        % steering 효과 기록
        steer_effect(k) = sin(alpha2 * tau_S_now);

        f1 = derivatives(state, tau_T_delayed, tau_S_now, ...
            Xu,Xuu,alpha1,alpha2,Yv,Yr,Yvv,Nr,Nv,Nrr,alpha3,alpha4);
        f2 = derivatives(state+0.5*dt*f1, tau_T_delayed, tau_S_now, ...
            Xu,Xuu,alpha1,alpha2,Yv,Yr,Yvv,Nr,Nv,Nrr,alpha3,alpha4);
        f3 = derivatives(state+0.5*dt*f2, tau_T_delayed, tau_S_now, ...
            Xu,Xuu,alpha1,alpha2,Yv,Yr,Yvv,Nr,Nv,Nrr,alpha3,alpha4);
        f4 = derivatives(state+dt*f3, tau_T_delayed, tau_S_now, ...
            Xu,Xuu,alpha1,alpha2,Yv,Yr,Yvv,Nr,Nv,Nrr,alpha3,alpha4);

        state = state + (dt/6)*(f1+2*f2+2*f3+f4);

        u_traj(k+1) = state(1);
        v_traj(k+1) = state(2);
        r_traj(k+1) = state(3);
    end
end

%% ===== dynamics =====
function dx = derivatives(state, tau_T_delayed, tau_S_now, ...
    Xu,Xuu,alpha1,alpha2,Yv,Yr,Yvv,Nr,Nv,Nrr,alpha3,alpha4)

    u=state(1); v=state(2); r=state(3);

    % throttle (지연된 값) & steering (즉시 값)
    F_t = tau_T_delayed.^2;

    % dynamics
    du = -Xu*u - Xuu*abs(u)*u + alpha1*F_t*cos(alpha2*tau_S_now);
    dv = -Yv*v - Yr*r - Yvv*abs(v)*v + alpha3*F_t*sin(alpha2*tau_S_now);
    dr = -Nr*r - Nv*v - Nrr*abs(r)*r - alpha4*F_t*sin(alpha2*tau_S_now);

    dx = [du,dv,dr];
end

%% ===== Helper: params struct → vector =====
function xvec = pack_params(p)
    xvec = zeros(1,12);
    xvec(1)=log(p.Xu); xvec(2)=log(p.Xuu);
    xvec(3)=log(p.alpha1); xvec(4)=-log(3/p.alpha2-1);
    xvec(5)=log(p.Yv); xvec(6)=log(p.Yr); xvec(7)=log(p.Yvv);
    xvec(8)=log(p.Nr); xvec(9)=log(p.Nv); xvec(10)=log(p.Nrr);
    xvec(11)=log(p.alpha3); xvec(12)=log(p.alpha4);
end

%% ===== Input conversion =====
function pwm1 = convertThrustToPwm(thrust)
    if thrust <= 1500, pwm1=(thrust-1450)*0.26;
    else, pwm1=(thrust-1550)*0.26; end
    pwm1 = max(pwm1,1.3);
end

function steer = convertSteeringToPwm(pwm)
    if pwm >= 2000
        steer = 300;
    elseif pwm > 1550 && pwm < 2000
        steer = (pwm - 1550)/1.6667;
    elseif pwm >= 1000 && pwm <= 1450
        steer = (pwm - 1450)/1.6667;
    elseif pwm <= 1000
        steer = -300;
    elseif pwm >= 1450 && pwm <= 1550
        steer = 0;        
    else
        steer = 0;
    end
end
