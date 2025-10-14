clc; clear; close all;

%% 공통 설정
dt = 0.1;
start_idx = 7330;
end_idx   = 15000;

files  = {'J1.csv'};
labels = {'Nominal MPC'};
styles = {'-'};

% ---- 스파이크 필터 설정 (속력 계산용) ----
phys_vmax    = 20;
spike_sigma  = 6;
median_win   = 5;

% ---- 플롯 스무딩 설정 (위치 오차 e(t) 시계열에 적용) ----
smooth.enable  = true;
smooth.method  = 'movmean';   % <- 오타 수정: movmean
smooth.fc_hz   = 0.01;        % (butter 사용시)
smooth.order   = 2;
smooth.win_sec = 10.0;
plot_raw_faint = false;

% ---- 시각화 파라미터 (여기서만 조절하세요) ----
viz.colors  = {[1 0 0], [0.2 0.8 0.2], [0 0 1]};  % r, orange, blue
viz.lineW   = 2.5;          % 선 굵기
viz.font.axis   = 12;       % 축 눈금 글씨
viz.font.label  = 14;       % xlabel, ylabel
viz.font.legend = 10;       % 범례
viz.font.title  = 16;       % 제목
viz.font.name   = 'Arial';  % 폰트 이름 (원하면 변경)

%% 계산 및 플로팅 (위치 오차)
results = cell(1, numel(files));
figure('Color','w'); hold on; grid on;
set(gca,'FontName',viz.font.name,'FontSize',viz.font.axis);  % 축 폰트/크기
t_end = 0;

for k = 1:numel(files)
    [t, err, stats] = compute_err(files{k}, start_idx, end_idx, dt, ...
                                  phys_vmax, spike_sigma, median_win);
    results{k}.t = t; results{k}.err = err; results{k}.stats = stats;

    % 스무딩 (플롯에만 적용)
    if smooth.enable
        err_s = smooth_series(err, dt, smooth);
    else
        err_s = err;
    end

    if plot_raw_faint
        plot(t, err, ':', 'Color',[0.6 0.6 0.6], 'LineWidth', 0.8, ...
             'DisplayName', [labels{k} ' (raw)']);
    end

    % 색상 + 굵기 지정
    plot(t, err_s, styles{k}, 'Color', viz.colors{k}, ...
         'LineWidth', viz.lineW, 'DisplayName', labels{k});

    fprintf(['%s: mean=%.3f m, RMSE=%.3f m, p95=%.3f m, max=%.3f m, ' ...
             'Vel.RMSE=%.3f m/s (N=%d)\n'], ...
        labels{k}, stats.mean, stats.rmse, stats.p95, stats.max, stats.vrmse, numel(err));

    t_end = max(t_end, t(end));
end

xlabel('Time [s]', 'FontName',viz.font.name, 'FontSize',viz.font.label);
ylabel('Tracking error [m]', 'FontName',viz.font.name, 'FontSize',viz.font.label);
% title('Tracking Error vs Time', 'FontName',viz.font.name, 'FontSize',viz.font.title);

lgd = legend('Location','best'); set(lgd,'Interpreter','none');
set(lgd,'FontName',viz.font.name,'FontSize',viz.font.legend);

xlim([0, t_end]);
box on;
set(gca,'LineWidth',1.5);

%% (옵션) 속도 오차 시계열 플롯
% figure('Color','w'); hold on; grid on;
% set(gca,'FontName',viz.font.name,'FontSize',viz.font.axis);
% for k = 1:numel(files)
%     plot(results{k}.stats.t_vel, abs(results{k}.stats.dv_series), ...
%          'LineWidth', viz.lineW-1, 'Color', viz.colors{k}, 'DisplayName', labels{k});
% end
% xlabel('Time [s]','FontName',viz.font.name,'FontSize',viz.font.label);
% ylabel('|v - v_{ref}| [m/s]','FontName',viz.font.name,'FontSize',viz.font.label);
% lgd2 = legend('Location','best'); set(lgd2,'Interpreter','none');
% set(lgd2,'FontName',viz.font.name,'FontSize',viz.font.legend);

%% ======== 로컬 함수들은 파일 맨 끝에 둡니다 ========
function [t, err, stats] = compute_err(fname, sidx, eidx, dt, phys_vmax, spike_sigma, median_win)
    T = readtable(fname);
    need = {'x','y','ref_x','ref_y'};
    if ~all(ismember(need, T.Properties.VariableNames))
        error('File %s must contain columns: x, y, ref_x, ref_y', fname);
    end

    x = T.x; y = T.y; refx = T.ref_x; refy = T.ref_y;

    N = height(T);
    sidx = max(1, sidx);
    eidx = min(N, eidx);
    if eidx < sidx
        error('Invalid range: start_idx (%d) > end_idx (%d) for %s', sidx, eidx, fname);
    end
    idx = sidx:eidx;

    dx  = x(idx) - refx(idx);
    dy  = y(idx) - refy(idx);
    err = sqrt(dx.^2 + dy.^2);

    t = (0:numel(idx)-1) * dt;

    xi    = x(idx);      yi    = y(idx);
    refxi = refx(idx);   refyi = refy(idx);

    v  = robust_speed(xi, yi, dt, phys_vmax, spike_sigma, median_win);
    vr = robust_speed(refxi, refyi, dt, phys_vmax, spike_sigma, median_win);

    L   = min(numel(v), numel(vr));
    v   = v(1:L);  vr = vr(1:L);
    ok  = ~isnan(v) & ~isnan(vr);

    dv              = v(ok) - vr(ok);
    stats.vrmse     = sqrt(mean(dv.^2));
    stats.dv_series = dv;
    stats.t_vel     = t(2:1+numel(v));

    stats.rmse = sqrt(mean(err.^2));
    stats.mean = mean(err);
    stats.max  = max(err);
    try
        stats.p95 = prctile(err, 95);
    catch
        es = sort(err);
        stats.p95 = es(max(1, round(0.95*numel(es))));
    end
end

function v = robust_speed(xi, yi, dt, phys_vmax, spike_sigma, median_win)
    dx = xi(2:end) - xi(1:end-1);
    dy = yi(2:end) - yi(1:end-1);
    v_raw = hypot(dx, dy) / dt;

    if isempty(v_raw), v = v_raw; return; end
    mask = false(size(v_raw));

    if isfinite(phys_vmax), mask = mask | (v_raw > phys_vmax); end

    medv = median(v_raw, 'omitnan');
    madv = median(abs(v_raw - medv), 'omitnan');
    if madv == 0, madv = eps; end
    z = abs(v_raw - medv) / (1.4826 * madv + eps);
    mask = mask | (z > spike_sigma);

    v_filt = v_raw; v_filt(mask) = NaN;
    v_filt = fillmissing(v_filt, 'linear', 'EndValues', 'nearest');
    if any(isnan(v_filt)), v_filt(isnan(v_filt)) = medv; end

    if median_win > 1
        try
            v_filt = movmedian(v_filt, median_win, 'omitnan');
        catch
            v_filt = movmean(v_filt, median_win, 'omitnan');
        end
    end
    v = v_filt;
end

function y = smooth_series(x, dt, smooth)
    if ~smooth.enable || numel(x) < 5
        y = x; return;
    end
    switch lower(smooth.method)
        case 'movmean'
            w = max(1, round(smooth.win_sec/dt));
            if mod(w,2)==0, w = w+1; end
            y = movmean(x, w, 'omitnan');
        case 'butter'
            fs = 1/dt; Wn = max(min(smooth.fc_hz/(fs/2),0.999),0.001);
            if exist('butter','file')==2 && exist('filtfilt','file')==2
                [b,a] = butter(smooth.order, Wn, 'low');
                try
                    y = filtfilt(b,a,double(x(:))); y = reshape(y,size(x));
                catch
                    w = max(1, round(smooth.win_sec/dt));
                    if mod(w,2)==0, w = w+1; end
                    y = movmean(x, w, 'omitnan');
                end
            else
                w = max(1, round(smooth.win_sec/dt));
                if mod(w,2)==0, w = w+1; end
                y = movmean(x, w, 'omitnan');
            end
        otherwise
            y = x;
    end
end
