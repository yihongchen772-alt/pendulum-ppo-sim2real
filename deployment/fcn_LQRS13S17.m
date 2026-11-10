function [pwm, x_out, v_out, theta_out, omega_out, amean_out] = fcn_LQRS13S17(cartpos_counts, rodAngle, cartVel_ms, rodVel_rads, lqr_pwm)
%#codegen
% =========================================================================
% 原厂 LQR 块、S13b、S17 同一次运行轮流跑，都不开外环（上机用；实验单_S17同次对照，11/09）。
% 整段替换 RL_Controller 里的代码。函数名不用改（11/05 的模型里保留了 fcn_LQRS12S8，照样编译）。
% 5 个输入、6 个输出，和 fcn_LQRS12S8 完全一样。第 5 个输入 lqr_pwm 接 Saturation1，11/05 存的模型已经接好。
% 第 6 路输出 amean_out = 平均读数角（rad，一阶低通，时间常数 5 s）。
%
% 【原厂块】保持原样，不加低通和积分（11/06 用户决定）。
% 【S13b】S12 带 25 Hz 输出低通续训，4 输入。【S17】第 320 局，5 输入：多一个小车位置积分 I。
%   两个网络都吃低通：y += a(mu*1000 - y)，a = 1 - exp(-2π·25·DT)；脉冲和探测加在低通之后，和训练环境一样。
%   进入网络段时低通从当前 PWM 接着走，不跳变。
%   I = clamp(I + x*DT/T_I, ±I_MAX)，T_I 0.5 s，I_MAX 4，每段开始清零，只在 S17 段内更新。
% 【两种跑法，程序一样】开跑前改 RodOffset 就行，不用重新编译：
%   校零遍：RodOffset = 校零值。零偏遍：校零值 + 13（约 −20 mrad）或 校零值 − 13（约 +20 mrad）。
%   第 1 段 40–85 s 的平均读数超过 12 mrad 就算零偏遍，这一遍 S13b 不加脉冲和探测（没有积分，小车已离中心约 140 mm）。
%   校零遍的残差不超过 ±8 mrad（超过要重新校零），零偏遍约 ±20 mrad，所以 12 mrad 分得开。
% 【计时】从 |alpha| < 0.15 rad 连续保持 1 s 起算，扶正之前是原厂块，不加扰动。
% 【时间表】单位 s，和 fcn_LQRS12S8 相同：
%   1 [   0, 150)  原厂  第 90/110/130 s 加 +/-/+ 脉冲（500 PWM × 0.1 s）
%   2 [ 150, 300)  S13b  同上（零偏遍不加）
%   3 [ 300, 450)  S17   同上
%   4 [ 450, 600)  原厂  第 90/110/130 s 加 -/+/- 脉冲
%   5 [ 600, 750)  S13b  同上（零偏遍不加）
%   6 [ 750, 900)  S17   同上
%   7 [ 900, 974)  原厂  静置 10 s 后步进正弦探测：2/2.5/3/3.5/3.96/4.5/5/6 Hz 各 8 s，150 PWM
%   8 [ 974,1048)  S13b  同上（零偏遍只静置）
%   9 [1048,1122)  S17   同上
%  10 [1122,1196)  原厂  同样的探测，75 PWM（11/07 加：v1 里 S17 的 |S| 随幅值变化大，75 PWM 时原厂块也不贴限幅）
%  11 [1196,1270)  S13b  同上（零偏遍只静置）
%  12 [1270,1344)  S17   同上
%   1344 s 以后     原厂，可以停
% 统计窗：1–6 段每段 40–85 s。
% 仿真验证：lqrs13s17_sim.m（v1 环境里逐步调用本文件）。
% =========================================================================
persistent net13 net17 isLoaded started n_hold t_bal a_mean a_sum n_sum y I_x seg_old no13
if isempty(isLoaded)
    net13 = coder.loadDeepLearningNetwork('actor_net_S13b_deploy.mat', 'actorNet');
    net17 = coder.loadDeepLearningNetwork('actor_net_S17_deploy.mat',  'actorNet');
    isLoaded = true;
    started  = false;
    n_hold   = 0;
    t_bal    = 0.0;
    a_mean   = 0.0;
    a_sum    = 0.0;
    n_sum    = 0.0;
    y        = 0.0;
    I_x      = 0.0;
    seg_old  = 1;
    no13     = false;
end

MAX_ACTION  = 1000.0;
X_OFFSET    = 0.0;
OMEGA_SCALE = 360/2048;   % 勿改
V_SIGN      = -1.0;       % 勿改
DT          = 0.004;

% ---- 时间表与扰动 ----
T_SEG    = 150.0;   % @T_SEG     轮流段每段秒数
N_ABC    = 6;       % @N_ABC     轮流段个数（原厂 S13b S17 原厂 S13b S17）
T_P      = 74.0;    % @T_P       每个探测段秒数（静置 10 s + 探测 64 s）
KICK_AMP = 500.0;   % @KICK_AMP  脉冲幅度（PWM）
KICK_DUR = 0.100;   % @KICK_DUR  脉冲宽度（s）
P_AMP    = 150.0;   % @P_AMP     第一轮探测幅度（PWM）
P_AMP2   = 75.0;    % @P_AMP2    第二轮探测幅度（PWM）
TH_START = 0.15;    % @TH_START  rad；扶正门限
N_HOLD   = 250;     % @N_HOLD    扶正条件要连续保持的步数（1 s）
A_MEAN   = 0.0008;  % @A_MEAN    平均读数角的低通系数（时间常数 5 s）
TH_OFF   = 0.012;   % @TH_OFF    rad；第 1 段 40–85 s 的平均读数超过它就算零偏遍
A_LPF    = 1 - exp(-2 * pi * 25 * DT);   % @A_LPF  25 Hz 输出低通，和训练一样
T_I      = 0.5;     % @T_I       积分时间，和训练一样
I_MAX    = 4.0;     % @I_MAX     积分限幅，和训练一样

x     = cartpos_counts - X_OFFSET;
alpha = rodAngle * (pi / 180);
v     = V_SIGN      * cartVel_ms;
omega = OMEGA_SCALE * rodVel_rads;

% ---- 计时：扶正并连续保持 1 s 后才开始 ----
if ~started
    if abs(alpha) < TH_START
        n_hold = n_hold + 1;
    else
        n_hold = 0;
    end
    if n_hold >= N_HOLD
        started = true;
        a_mean  = alpha;
    end
end
if started
    t_bal  = t_bal + DT;
    a_mean = a_mean + A_MEAN * (alpha - a_mean);
    if t_bal >= 40.0 && t_bal < 85.0
        a_sum = a_sum + alpha;
        n_sum = n_sum + 1.0;
    end
end

% ---- 当前段：1–6 = 轮流段，7–9 = 150 PWM 探测段，10–12 = 75 PWM 探测段，13 = 结束 ----
if t_bal < N_ABC * T_SEG
    seg = floor(t_bal / T_SEG) + 1;
    ts  = t_bal - (seg - 1) * T_SEG;
elseif t_bal < N_ABC * T_SEG + 6 * T_P
    seg = N_ABC + 1 + floor((t_bal - N_ABC * T_SEG) / T_P);
    ts  = t_bal - N_ABC * T_SEG - (seg - N_ABC - 1) * T_P;
else
    seg = 13; ts = 0.0;
end

% ---- 进入新的一段：积分清零；第 2 段开始时判断这一遍是不是零偏遍 ----
if seg ~= seg_old
    seg_old = seg;
    I_x = 0.0;
    if seg == 2
        no13 = abs(a_sum) > TH_OFF * n_sum;
    end
end

% ---- 这一段用哪个控制器：1 = 原厂，2 = S13b，3 = S17（扶正之前 seg = 1，是原厂）----
if seg <= N_ABC
    ctl = mod(seg - 1, 3) + 1;
elseif seg <= N_ABC + 6
    ctl = mod(seg - N_ABC - 1, 3) + 1;
else
    ctl = 1;
end

% ---- 控制器 ----
if ctl == 1
    y   = lqr_pwm;             % 原厂块原样转发（它自己的速度滤波、F/Volt、小信号 ×1.2、限幅都在块里）
    pwm = lqr_pwm;
else
    state = [x; v / 5.0; alpha; omega / 10.0];
    if ctl == 3
        I_x = max(min(I_x + x * DT / T_I, I_MAX), -I_MAX);
        dl5 = dlarray(single([state; I_x]), 'CB');
        out_actor = extractdata(predict(net17, dl5));
    else
        dl4 = dlarray(single(state), 'CB');
        out_actor = extractdata(predict(net13, dl4));
    end
    mu  = tanh(double(out_actor(1)));
    y   = y + A_LPF * (mu * MAX_ACTION - y);
    pwm = y;
end

% ---- 这一段要不要加扰动：零偏遍的 S13b 段什么都不加 ----
disturb = started && ~(ctl == 2 && no13);

% ---- 可重复的扰动脉冲（轮流段每段 3 次：1–3 段为 +/-/+，4–6 段为 -/+/-）----
extra = 0.0;
if disturb && seg <= N_ABC
    kk = 0;
    if     ts >= 90.0  && ts < 90.0  + KICK_DUR, kk = 1;
    elseif ts >= 110.0 && ts < 110.0 + KICK_DUR, kk = 2;
    elseif ts >= 130.0 && ts < 130.0 + KICK_DUR, kk = 3;
    end
    if kk > 0
        sgn = 1.0;
        if kk == 2, sgn = -1.0; end
        if seg > 3, sgn = -sgn; end
        extra = sgn * KICK_AMP;
    end
end

% ---- 步进正弦探测（7、10 原厂，8、11 S13b，9、12 S17）：8 个频点各 8 s，每段开头 0.5 s 爬升 ----
if disturb && seg > N_ABC && seg <= N_ABC + 6
    tp = ts - 10.0;
    if tp >= 0.0 && tp < 64.0
        k8 = floor(tp / 8.0) + 1;
        fp = 2.0;
        if     k8 == 2, fp = 2.5;
        elseif k8 == 3, fp = 3.0;
        elseif k8 == 4, fp = 3.5;
        elseif k8 == 5, fp = 3.96;
        elseif k8 == 6, fp = 4.5;
        elseif k8 == 7, fp = 5.0;
        elseif k8 == 8, fp = 6.0;
        end
        tk = tp - (k8 - 1) * 8.0;
        amp = P_AMP;
        if seg > N_ABC + 3, amp = P_AMP2; end
        extra = amp * min(tk / 0.5, 1.0) * sin(2 * pi * fp * tk);
    end
end

pwm = pwm + extra;
if pwm >  MAX_ACTION, pwm =  MAX_ACTION; end
if pwm < -MAX_ACTION, pwm = -MAX_ACTION; end

x_out = x; v_out = v; theta_out = alpha; omega_out = omega; amean_out = a_mean;
end
