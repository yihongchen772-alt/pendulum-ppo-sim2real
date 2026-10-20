function [pwm, x_out, v_out, theta_out, omega_out, aref_out] = fcn_S8S12_ABAB(cartpos_counts, rodAngle, cartVel_ms, rodVel_rads, lqr_pwm) %#ok<INUSD>
%#codegen
% =========================================================================
% S8 对 S12（上机用）：一个函数块里装两个网络，按时间表自动切换，外环全程开。
% 贴进 RL_Controller 的 MATLAB Function 块时，只把第一行的函数名 fcn_S8S12_ABAB 改回 fcn，其余逐字不动。
% 接口与 10/02 相同（5 个输入、6 个输出），第 5 个输入 lqr_pwm 接着即可、不使用，所以不用改线。
%
% 【两个网络】S8 = actor_net_S8_deploy.mat（基线，9/21 起一直在用）；
%            S12 = actor_net_S12_deploy.mat（在校准仿真 v1 里训练，train_rl_S12.m，ep08000）。
% 两者的观测换算完全相同，与 fcn_S8_OLc.m 逐字一致。
%
% 【外环】与 fcn_S8_OLc.m 相同（K_I、低通、死区不变），只改两处，原因是 10/02 零偏达 -33/-44 mrad：
%   ① AREF_MAX 0.025 -> 0.050 rad：旧限幅小于 10/02 的零偏，会积分到头、居中不完全；
%   ② 冻结条件 |alpha| -> |alpha - aref|：零偏大时读数本身就常超过 0.06，积分会被频繁冻结。
% 外环从扶正起一直开，切换网络时不清零（aref 估的是零偏，与用哪个网络无关）。
%
% 【时间表】从第一次 |alpha - aref| < 0.05 rad（扶正）起算，单位 s：
%   A1 [  0,120)  S8    第 100 s 加 +脉冲
%   B1 [120,240)  S12   第 220 s 加 +脉冲
%   A2 [240,360)  S8    第 340 s 加 -脉冲
%   B2 [360,480)  S12   第 460 s 加 -脉冲
%   P1 [480,554)  S8    静置 10 s 后步进正弦探测：2/2.5/3/3.5/3.96/4.5/5/6 Hz 各 8 s，150 PWM
%   P2 [554,628)  S12   同上
%   628 s 以后     S8，可以停
% 第 6 路输出 aref_out = alpha_ref（rad）。
% 仿真验证：s12_ab_sim.m（v1 环境里逐步调用本文件跑完整时间表）。
% =========================================================================
persistent net8 net12 isLoaded started t_bal x_f aref
if isempty(isLoaded)
    net8  = coder.loadDeepLearningNetwork('actor_net_S8_deploy.mat',  'actorNet');
    net12 = coder.loadDeepLearningNetwork('actor_net_S12_deploy.mat', 'actorNet');
    isLoaded = true;
    started  = false;
    t_bal    = 0.0;
    x_f      = 0.0;
    aref     = 0.0;
end

MAX_ACTION  = 1000.0;
X_OFFSET    = 0.0;
OMEGA_SCALE = 360/2048;   % 勿改
V_SIGN      = -1.0;       % 勿改
DT          = 0.004;

% ---- 外环参数 ----
K_I      = 0.015;   % @K_I       rad/(m*s)；正号：小车偏负 -> alpha_ref 往负走
A_LPF    = 0.0050;  % @A_LPF     小车位置一阶低通系数（约 0.2 Hz）
DEADBAND = 0.003;   % @DEADBAND  m；|低通 x| 小于它不积分，防静摩擦来回蹭
AREF_MAX = 0.050;   % @AREF_MAX  rad；alpha_ref 限幅（约 2.9 度；fcn_S8_OLc 为 0.025）
FREEZE   = 0.06;    % @FREEZE    rad；|alpha - aref| 超过它暂停积分（被推、大摆动时）
% ---- 时间表与扰动 ----
T_SEG    = 120.0;   % @T_SEG     ABAB 每段秒数
T_P      = 74.0;    % @T_P       每个探测段秒数（静置 10 s + 探测 64 s）
KICK_AT  = 100.0;   % @KICK_AT   段内第几秒加脉冲
KICK_AMP = 500.0;   % @KICK_AMP  脉冲幅度（PWM）
KICK_DUR = 0.100;   % @KICK_DUR  脉冲宽度（s）
P_AMP    = 150.0;   % @P_AMP     探测幅度（PWM）

x     = cartpos_counts - X_OFFSET;
alpha = rodAngle * (pi / 180);
v     = V_SIGN      * cartVel_ms;
omega = OMEGA_SCALE * rodVel_rads;

% ---- 计时：扶正后才开始 ----
if ~started && abs(alpha - aref) < 0.05
    started = true;
end
if started
    t_bal = t_bal + DT;
end

% ---- 当前段：1–4 = ABAB，5 = P1，6 = P2，7 = 结束 ----
if t_bal < 4 * T_SEG
    seg = floor(t_bal / T_SEG) + 1;
    ts  = t_bal - (seg - 1) * T_SEG;
elseif t_bal < 4 * T_SEG + T_P
    seg = 5; ts = t_bal - 4 * T_SEG;
elseif t_bal < 4 * T_SEG + 2 * T_P
    seg = 6; ts = t_bal - 4 * T_SEG - T_P;
else
    seg = 7; ts = 0.0;
end
use12 = started && (seg == 2 || seg == 4 || seg == 6);

% ---- 慢速外环（扶正后一直开）----
x_f = x_f + A_LPF * (x - x_f);
if started && abs(alpha - aref) < FREEZE && abs(x_f) > DEADBAND
    aref = aref + K_I * x_f * DT;
    if aref >  AREF_MAX, aref =  AREF_MAX; end
    if aref < -AREF_MAX, aref = -AREF_MAX; end
end

% ---- 网络（观测换算与 fcn_S8_OLc 逐字相同，角度减去 alpha_ref）----
state   = [x; v / 5.0; alpha - aref; omega / 10.0];
dlState = dlarray(single(state), 'CB');
if use12
    out_actor = extractdata(predict(net12, dlState));
else
    out_actor = extractdata(predict(net8, dlState));
end
mu  = tanh(double(out_actor(1)));
pwm = mu * MAX_ACTION;

% ---- 可重复的扰动脉冲（ABAB 每段一次）----
extra = 0.0;
if started && seg <= 4 && ts >= KICK_AT && ts < KICK_AT + KICK_DUR
    if seg <= 2
        extra = KICK_AMP;
    else
        extra = -KICK_AMP;
    end
end

% ---- 步进正弦探测（P1、P2）：8 个频点各 8 s，每段开头 0.5 s 爬升 ----
if started && (seg == 5 || seg == 6)
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
        extra = P_AMP * min(tk / 0.5, 1.0) * sin(2 * pi * fp * tk);
    end
end

pwm = pwm + extra;
if pwm >  MAX_ACTION, pwm =  MAX_ACTION; end
if pwm < -MAX_ACTION, pwm = -MAX_ACTION; end

x_out = x; v_out = v; theta_out = alpha; omega_out = omega; aref_out = aref;
end
