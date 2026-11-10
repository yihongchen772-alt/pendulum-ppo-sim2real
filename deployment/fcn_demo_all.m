function [pwm, x_out, v_out, theta_out, omega_out, amean_out] = fcn_demo_all(cartpos_counts, rodAngle, cartVel_ms, rodVel_rads, lqr_pwm)
%#codegen
% =========================================================================
% 答辩演示（11/10 上机，五遍数据跑完以后才用）：V5 手推 + V6 S8 对 ABL-E，合成一遍，只编译一次。
% 由 fcn_LQRS13S17.m 和 fcn_demo_S8E.m 合并：接口相同（5 入 6 出，第 5 个输入接 Saturation1），
% 不加脉冲、不做探测，只按时间切控制器；推由人手做。这一遍只拍视频，不进主分析。
% 贴进 RL_Controller 时整段替换，函数名不用改。
% 【时间表】从扶正算起（|alpha| < 0.15 rad 连续 1 s 开始计时），单位 s：
%   1 [  0,  60)  原厂块   手推
%   2 [ 60, 120)  S13b     手推（25 Hz 低通，同 fcn_LQRS13S17）
%   3 [120, 180)  S17      手推（25 Hz 低通 + 积分输入，同 fcn_LQRS13S17；本段开始积分清零）
%   4 [180, 240)  S8       手推（actor_net_S8_deploy.mat，不加低通，同第 6 章）
%   5 [240, 300)  S8       不推，静置
%   6 [300, 360)  ABL-E    不推（actor_net_ABL_E_s102_best.mat，9/21 上过机：静置约 3 Hz 振荡）
%   360 s 以后    原厂块，可以停
% 【安全】网络段里摆角偏离平均读数或小车超过门限，就切回原厂块，直到这一段结束（下一段照常切）。
%   手推段（2–4）门限 0.30 rad / 0.28 m（手推会把摆角推到 10° 以上）；不推的段（5、6）0.20 rad / 0.25 m，同 fcn_demo_S8E。
%   硬件联锁是 30°、340 mm。
% 仿真验证：demo_all_sim.m（v1 里逐步调用本文件，含模拟手推）。
% =========================================================================
persistent net8 net13 net17 netE isLoaded started n_hold t_bal a_mean y I_x seg_old aborted
if isempty(isLoaded)
    net8  = coder.loadDeepLearningNetwork('actor_net_S8_deploy.mat',       'actorNet');
    net13 = coder.loadDeepLearningNetwork('actor_net_S13b_deploy.mat',     'actorNet');
    net17 = coder.loadDeepLearningNetwork('actor_net_S17_deploy.mat',      'actorNet');
    netE  = coder.loadDeepLearningNetwork('actor_net_ABL_E_s102_best.mat', 'actorNet');
    isLoaded = true;
    started  = false;
    n_hold   = 0;
    t_bal    = 0.0;
    a_mean   = 0.0;
    y        = 0.0;
    I_x      = 0.0;
    seg_old  = 0;
    aborted  = false;
end

MAX_ACTION  = 1000.0;
X_OFFSET    = 0.0;
OMEGA_SCALE = 360/2048;   % 勿改
V_SIGN      = -1.0;       % 勿改
DT          = 0.004;

T_SEG    = 60.0;    % @T_SEG     每段秒数
N_SEG    = 6;       % @N_SEG     段数
TH_START = 0.15;    % @TH_START  rad；扶正门限
N_HOLD   = 250;     % @N_HOLD    扶正条件要连续保持的步数（1 s）
A_MEAN   = 0.0008;  % @A_MEAN    平均读数角的低通系数（时间常数 5 s）
TH_TAP   = 0.30;    % @TH_TAP    rad；手推段切回门限
X_TAP    = 0.28;    % @X_TAP     m；手推段切回门限
TH_ABORT = 0.20;    % @TH_ABORT  rad；不推的段切回门限
X_ABORT  = 0.25;    % @X_ABORT   m；不推的段切回门限
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
end

% ---- 当前段：1–6，7 = 结束；进入新的一段时积分清零、安全切回复位 ----
seg = 7;
if t_bal < N_SEG * T_SEG
    seg = floor(t_bal / T_SEG) + 1;
end
if seg ~= seg_old
    seg_old = seg;
    I_x     = 0.0;
    aborted = false;
end

% ---- 这一段用哪个控制器：1 = 原厂，2 = S13b，3 = S17，4 = S8，5 = ABL-E（扶正之前是原厂）----
ctl = 1;
if started
    if     seg == 2, ctl = 2;
    elseif seg == 3, ctl = 3;
    elseif seg == 4 || seg == 5, ctl = 4;
    elseif seg == 6, ctl = 5;
    end
end
th_lim = TH_ABORT; x_lim = X_ABORT;
if seg >= 2 && seg <= 4
    th_lim = TH_TAP; x_lim = X_TAP;
end
if ctl > 1 && (aborted || abs(alpha - a_mean) > th_lim || abs(x) > x_lim)
    aborted = true;
    ctl = 1;
end

% ---- 控制器 ----
state = [x; v / 5.0; alpha; omega / 10.0];
if ctl == 1
    pwm = lqr_pwm;             % 原厂块原样转发
    y   = pwm;
elseif ctl == 2 || ctl == 3
    if ctl == 3
        I_x = max(min(I_x + x * DT / T_I, I_MAX), -I_MAX);
        dl5 = dlarray(single([state; I_x]), 'CB');
        out_actor = extractdata(predict(net17, dl5));
    else
        dl4 = dlarray(single(state), 'CB');
        out_actor = extractdata(predict(net13, dl4));
    end
    mu  = tanh(double(out_actor(1)));
    y   = y + A_LPF * (mu * MAX_ACTION - y);   % 低通从当前 PWM 接着走
    pwm = y;
else
    dl4 = dlarray(single(state), 'CB');
    if ctl == 4
        out_actor = extractdata(predict(net8, dl4));
    else
        out_actor = extractdata(predict(netE, dl4));
    end
    mu  = tanh(double(out_actor(1)));
    pwm = mu * MAX_ACTION;
    y   = pwm;
end

if pwm >  MAX_ACTION, pwm =  MAX_ACTION; end
if pwm < -MAX_ACTION, pwm = -MAX_ACTION; end

x_out = x; v_out = v; theta_out = alpha; omega_out = omega; amean_out = a_mean;
end
