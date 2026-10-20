% 上机前仿真：在校准仿真 v1 里逐步调用上机文件 fcn_S8S12_ABAB.m，原样跑完整时间表（约 640 s）。
% 验证的就是部署代码本身（接口换算、外环、切换、脉冲、探测），同时给出事前预测的数字。
% 零偏取 -20 mrad（9/24–9/28 的量级）和 -44 mrad（10/02 的最大值），各 2 个种子。
% 用法：matlab -batch "s12_ab_sim"；结果写 logs/S12_hw/sim_prediction.txt
cd(fileparts(mfilename('fullpath')));
P = v1_params(); DT = P.DT; TS = 120; TP = 74;
FS = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0];
if ~exist(fullfile('logs','S12_hw'),'dir'), mkdir(fullfile('logs','S12_hw')); end
fid = fopen(fullfile('logs','S12_hw','sim_prediction.txt'),'w');
lg = @(varargin) [fprintf(varargin{:}), fprintf(fid, varargin{:})];
NM = {'A1 S8','B1 S12','A2 S8','B2 S12'};
for BA = [-0.020 -0.044]
    for sd = 1:2
        clear fcn_S8S12_ABAB
        rng(sd); th0 = (rand-0.5)*0.01;
        E = v1_env_init(P, th0, randi(numel(P.noise))-1, BA);
        N = round((4*TS + 2*TP + 10)/DT);
        L = zeros(5,N); n = N; t_start = NaN;
        for k = 1:N
            [o,E,al] = v1_env_observe(E,P);
            hw = [o(1), al*180/pi, -o(2)*5, o(4)*10/(360/2048)];   % cartpos(m)、rodAngle(度)、cartVel(V_SIGN=-1)、rodVel(OMEGA_SCALE)
            [pwm,~,~,~,~,aref] = fcn_S8S12_ABAB(hw(1), hw(2), hw(3), hw(4), 0.0);
            E = v1_env_step(E,P,pwm);
            L(:,k) = [E.s(1); E.s(3); al; pwm; aref];
            if isnan(t_start) && abs(al) < 0.05, t_start = (k-1)*DT; end
            if abs(E.s(3)) > 0.5 || abs(E.s(1)) > 0.33, n = k; break; end
        end
        L = L(:,1:n); t = (1:n)*DT - t_start - DT;     % 与 fcn 内的 t_bal 对齐
        X = L(1,:); TH = L(2,:); AL = L(3,:); U = L(4,:); AR = L(5,:);
        if BA == -0.020 && sd == 1   % 存一条给分析脚本自检：python3 s12_hw_analyze.py --sim logs/S12_hw/sim_run.mat
            R.t = (1:n)*DT; R.x = X; R.al = AL; R.aref = AR; R.pwm = U; save(fullfile('logs','S12_hw','sim_run.mat'), 'R');
        end
        lg('\n=== 零偏 %+.0f mrad，种子 %d：%s ===\n', BA*1000, sd, ternary(n==N,'跑完','**倒了**，第 %.0f s', t(end)));
        lg('段      θ std°  PWM std  ΔPWM   x 均值 mm  x std mm  aref mrad | 脉冲 θ 偏离°  恢复 s\n');
        for s = 1:4
            s0 = (s-1)*TS; m = t>=s0+50 & t<s0+95;
            if sum(m) < 1000, continue; end
            a = AL(m); u = U(m);
            pre = t>=s0+95 & t<s0+100; m3 = t>=s0+100 & t<s0+103; m10 = t>=s0+100 & t<s0+110;
            a0 = mean(AL(pre)); dth = rad2deg(max(abs(AL(m3)-a0)));
            ok = abs(AL(m10)-a0) <= 0.05; c = 0; rr = NaN; tm = t(m10);
            for i = 1:numel(ok), if ok(i), c = c+1; else, c = 0; end, if c >= 250, rr = tm(i-249)-(s0+100); break; end, end
            lg('%-7s %6.2f  %7.1f  %5.1f  %+9.1f  %8.1f  %+8.1f | %6.1f  %6.2f\n', NM{s}, rad2deg(std(a)), std(u), ...
                sqrt(mean(diff(u).^2)), 1000*mean(X(m)), 1000*std(X(m)), 1000*AR(find(m,1,'last')), dth, rr);
        end
        for p = 1:2
            p0 = 4*TS + (p-1)*TP + 10; Sv = nan(1,numel(FS));
            for i = 1:numel(FS)
                m = t >= p0+8*(i-1)+1 & t < p0+8*i;
                if sum(m) < 1000, continue; end
                tk = t(m) - (p0+8*(i-1)); pr = 150*sin(2*pi*FS(i)*tk); e = exp(-1j*2*pi*FS(i)*tk);
                Sv(i) = abs(sum((U(m)-mean(U(m))).*e)) / abs(sum(pr.*e));
            end
            mx = t >= p0 & t < p0+64;
            lg('P%d %-4s |S| %s  （探测期间小车 %+.0f…%+.0f mm）\n', p, ternary(p==1,'S8','S12'), sprintf('%5.2f',Sv), ...
                1000*min(X(mx)), 1000*max(X(mx)));
        end
    end
end
fclose(fid);

function s = ternary(c, a, b, varargin)
    if c, s = a; else, s = sprintf(b, varargin{:}); end
end
