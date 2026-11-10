% 上机前仿真：在校准仿真 v1 里逐步调用上机文件 fcn_LQRS13S17.m，原样跑完整时间表（1344 s）。
% 原厂 LQR 块按 lqrs12s8_sim.m 移植，算出的 PWM 接上机文件的第 5 个输入（真机上是 Saturation1）。
% 校零遍用零偏 0、±5、±10 mrad（校零后的残差），零偏遍用 ±20 mrad。
% 过程噪声 ×1 看安全，×0.4 看晃动（×0.4 是事后拟合，只用于看晃动）。
% v1 对原厂块的静置摆角高估约 80%，原厂段的摆角只作参考。
% 用法：matlab -batch "lqrs13s17_sim"；结果写 logs/S17_hw/sim_prediction.txt
cd(fileparts(mfilename('fullpath')));
P = v1_params(); DT = P.DT; TS = 150; TP = 74; NA = 6; TK = [90 110 130];
FS = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0];
K_LQR = [-38.7298 164.2110 -76.3417 41.0570];                      % 原厂 Q=diag([3000,5000,1,1])、R=2
Rm = 2.12; rmp = 0.013745; Kt = 0.08505; Kb = 0.11693; Beq = 30.0; Mt = 0.4719 + 0.1679;
b1 = Rm*rmp^2*Mt; b0 = Rm*rmp^2*Beq + Kt*Kb; a1 = Kt*rmp*Mt; a0 = Kt*rmp*Beq; KT = 2/DT;
FV = [b1*KT + b0, b0 - b1*KT, a0 - a1*KT, a1*KT + a0];             % F/Volt 的 Tustin 系数（与 s8_probe_sim.FV 相同）
OUT = fullfile('logs','S17_hw'); if ~exist(OUT,'dir'), mkdir(OUT); end
fid = fopen(fullfile(OUT,'sim_prediction.txt'),'w');
lg = @(varargin) [fprintf(varargin{:}), fprintf(fid, varargin{:})];
CTL = {'原厂','S13b','S17','原厂','S13b','S17','原厂','S13b','S17','原厂','S13b','S17'};
CASES = [0 1 1; 0 2 1; -0.005 1 1; 0.005 1 1; -0.010 1 1; 0.010 1 1; -0.020 1 1; 0.020 1 1; ...
         0 3 0.4; 0 4 0.4; -0.020 3 0.4; 0.020 3 0.4];   % [零偏 rad, 种子, 过程噪声倍数]
tic;
for c = 1:size(CASES,1)
    BA = CASES(c,1); sd = CASES(c,2); ds = CASES(c,3);
    clear fcn_LQRS13S17
    rng(sd); th0 = (rand-0.5)*0.01;
    E = v1_env_init(P, th0, randi(numel(P.noise))-1, BA);
    TEND = NA*TS + 6*TP + 12;
    N = round(TEND/DT);
    L = zeros(5,N); n = N; fu = 0; fy = 0;
    for k = 1:N
        [o,E,al,x_q] = v1_env_observe(E,P);
        uF = -(K_LQR * [x_q; al; o(2)*5; o(4)*10/P.OM]);            % 原厂块自己的 θ̇ 单位正确（不是 2 倍）
        uF = max(min(uF, 35), -35);
        fy = (FV(1)*uF + FV(2)*fu - FV(3)*fy) / FV(4); fu = uF;
        pl = fy*1000/12; if abs(pl) <= 100, pl = 1.2*pl; end
        pl = max(min(pl, 1000), -1000);
        hw = [o(1), al*180/pi, -o(2)*5, o(4)*10/(360/2048)];   % cartpos(m)、rodAngle(度)、cartVel(V_SIGN=-1)、rodVel(OMEGA_SCALE)
        [pwm,~,~,~,~,am] = fcn_LQRS13S17(hw(1), hw(2), hw(3), hw(4), pl);
        E = v1_env_step(E,P,pwm,ds);
        L(:,k) = [E.s(1); E.s(3); al; pwm; am];
        if abs(E.s(3)) > 0.5 || abs(E.s(1)) > 0.33, n = k; break; end
    end
    L = L(:,1:n); X = L(1,:); AL = L(3,:); U = L(4,:); AM = L(5,:);
    % 起点与 fcn 相同：|alpha| < 0.15 连续 250 步，第 250 步 t_bal = DT
    ok = abs(AL) < 0.15; cnt = 0; k0 = NaN;
    for k = 1:n, if ok(k), cnt = cnt+1; else, cnt = 0; end, if cnt >= 250, k0 = k; break; end, end
    t = ((1:n) - k0 + 1)*DT;
    if BA == 0 && sd == 1          % 给分析脚本自检：python3 s17_hw_analyze.py --sim logs/S17_hw/sim_run.mat
        R.t = (1:n)*DT; R.x = X; R.al = AL; R.am = AM; R.pwm = U; save(fullfile(OUT,'sim_run.mat'), 'R');
    elseif BA == -0.020 && sd == 1
        R.t = (1:n)*DT; R.x = X; R.al = AL; R.am = AM; R.pwm = U; save(fullfile(OUT,'sim_run_off.mat'), 'R');
    end
    lg('\n=== 零偏 %+.0f mrad，种子 %d，过程噪声 ×%.1f：%s ===\n', BA*1000, sd, ds, ternary(n==N,'跑完','**倒了或撞联锁**，第 %.0f s', t(end)));
    lg('段        θ std°  PWM std  ΔPWM   x 均值 mm  x std mm  慢晃 mm  读数均值 mrad  段内小车最远 mm | 切换后 10 s θ 最大偏离° | 3 次脉冲（θ 偏离° / 恢复 s / 小车最远 mm）\n');
    for s = 1:NA
        s0 = (s-1)*TS; m = t >= s0+40 & t < s0+85;
        if sum(m) < 1000, continue; end
        a = AL(m); u = U(m); ms = t >= s0 & t < s0+TS; xs = X(ms); [~,j] = max(abs(xs));
        m10 = t >= s0 & t < s0+10; sw = rad2deg(max(abs(AL(m10) - mean(AL(m)))));
        pk = '';
        for kk = 1:3
            tk = s0 + TK(kk);
            pre = t >= tk-5 & t < tk; m3 = t >= tk & t < tk+3; mr = t >= tk & t < tk+10;
            if sum(mr) < 2000, continue; end
            a0 = mean(AL(pre)); dth = rad2deg(max(abs(AL(m3)-a0)));
            okr = abs(AL(mr)-a0) <= 0.05; cc = 0; rr = NaN; tm = t(mr);
            for i = 1:numel(okr), if okr(i), cc = cc+1; else, cc = 0; end, if cc >= 250, rr = tm(i-249)-tk; break; end, end
            xr = X(mr); [~,jr] = max(abs(xr));
            pk = [pk sprintf('  %4.1f / %4.2f / %+4.0f', dth, rr, 1000*xr(jr))]; %#ok<AGROW>
        end
        lg('%d %-5s %6.2f  %7.1f  %5.1f  %+9.1f  %8.1f  %7.1f  %+13.1f  %+15.0f | %22.1f |%s\n', s, CTL{s}, rad2deg(std(a)), std(u), ...
            sqrt(mean(diff(u).^2)), 1000*mean(X(m)), 1000*std(X(m)), sway(X(m),DT), 1000*mean(a), 1000*xs(j), sw, pk);
    end
    for s = NA+1:NA+6
        p0 = NA*TS + (s-NA-1)*TP + 10; mx = t >= p0 & t < p0+64; amp = 150; if s > NA+3, amp = 75; end
        if sum(mx) < 1000, continue; end
        Sv = nan(1,numel(FS));
        for i = 1:numel(FS)
            m = t >= p0+8*(i-1)+1 & t < p0+8*i;
            if sum(m) < 1000, continue; end
            tk = t(m) - (p0+8*(i-1)); pr = amp*sin(2*pi*FS(i)*tk); e = exp(-1j*2*pi*FS(i)*tk);
            Sv(i) = abs(sum((U(m)-mean(U(m))).*e)) / abs(sum(pr.*e));
        end
        lg('%d %-5s %3d PWM |S| %s  （2/2.5/3/3.5/3.96/4.5/5/6 Hz；探测期间小车 %+.0f…%+.0f mm，PWM 贴限幅 %.0f%%）\n', s, CTL{s}, amp, ...
            sprintf('%5.2f',Sv), 1000*min(X(mx)), 1000*max(X(mx)), 100*mean(abs(U(mx)) >= 999));
    end
    lg('全程小车最远 %+.0f mm；（已用 %.0f s）\n', 1000*X(find(abs(X) == max(abs(X)), 1)), toc);
end
fclose(fid);

function s = ternary(c, a, b, varargin)
    if c, s = a; else, s = sprintf(b, varargin{:}); end
end

function s = sway(x, DT)
    % 0.1–0.5 Hz 小车晃动 mm（Hann + FFT，和 s16_land 同口径；窗只有 45 s，只作预测参考）
    x = 1000*(x - mean(x)); n = numel(x); h = 0.5 - 0.5*cos(2*pi*(0:n-1)/(n-1));
    Xf = abs(fft(x.*h)).^2 / (sum(h.^2)*n); f = (0:n-1)/(n*DT);
    s = sqrt(2*sum(Xf(f >= 0.1 & f <= 0.5)));
end
