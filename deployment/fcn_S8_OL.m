function [pwm, x_out, v_out, theta_out, omega_out, aref_out] = fcn_S8_OL(cartpos_counts, rodAngle, cartVel_ms, rodVel_rads)
%#codegen
% =========================================================================
% S8 + 慢速外环居中（上机用）。贴进 RL_Controller 的 MATLAB Function 块时，
% 只把第一行的函数名 fcn_S8_OL 改回 fcn，其余逐字不动。
%
% 【为什么】S8 是没有记忆的网络：只要用角度反馈，遇到未知的常值偏差
% （角度零偏、开机标定差）就一定留下稳态偏移。想压小它只能全频段抬 Kx，
% S10/S11 已证明那样抖动变大、居中还不变。积分器只在极低频起作用（内模原理）。
%
% 【做法】alpha_ref 随低通后的小车位置慢慢积分，喂给网络的角度 = alpha - alpha_ref。
% 稳态时 alpha_ref 收敛到角度零偏，小车回到中心。网络本身一个字不改。
%
% 【时间表】从第一次 |alpha| < 0.05 rad（扶正）起算，单位 s。一次运行做完 ABAB，
% 不用中途重新编译：
%   A1 [  0,120)  纯 S8                      第 100 s 加 +脉冲
%   B1 [120,240)  S8 + 外环（从 0 开始积分）  第 220 s 加 +脉冲
%   A2 [240,360)  纯 S8（外环清零）           第 340 s 加 -脉冲
%   B2 [360,480)  S8 + 外环（从 0 重新积分）  第 460 s 加 -脉冲
%   P  [480,570)  纯 S8；静置 20 s 后步进正弦探测（测 S8 的闭环灵敏度 Ms）
%   570 s 以后     纯 S8，可以停
% 第 6 路输出 aref_out = alpha_ref（rad），可在示波器上看它慢慢收敛。
% 参数由 ol_center_sim.m 在仿真里定（该脚本直接调用本文件做闭环仿真）。
% =========================================================================
persistent actorNet isLoaded started t_bal x_f aref
if isempty(isLoaded)
    actorNet = coder.loadDeepLearningNetwork('actor_net_S8_deploy.mat', 'actorNet');
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
AREF_MAX = 0.025;   % @AREF_MAX  rad；alpha_ref 限幅（约 1.4 度）
FREEZE   = 0.06;    % @FREEZE    rad；|alpha| 超过它暂停积分（被推、大摆动时）
% ---- 时间表与扰动 ----
T_SEG    = 120.0;   % @T_SEG     每段秒数
KICK_AT  = 100.0;   % @KICK_AT   段内第几秒加脉冲
KICK_AMP = 700.0;   % @KICK_AMP  脉冲幅度（PWM）
KICK_DUR = 0.100;   % @KICK_DUR  脉冲宽度（s）
P_AMP    = 150.0;   % @P_AMP     探测幅度（PWM）；9/07 用 200 时峰值处 34–47% 时间贴 ±1000，Ms 被削低，所以降到 150

x     = cartpos_counts - X_OFFSET;
alpha = rodAngle * (pi / 180);
v     = V_SIGN      * cartVel_ms;
omega = OMEGA_SCALE * rodVel_rads;

% ---- 计时：扶正后才开始 ----
if ~started && abs(alpha) < 0.05
    started = true;
end
if started
    t_bal = t_bal + DT;
end
seg = floor(t_bal / T_SEG) + 1;           % 1=A1 2=B1 3=A2 4=B2 5=P
ts  = t_bal - (seg - 1) * T_SEG;           % 段内时间
ol_on = started && (seg == 2 || seg == 4);

% ---- 慢速外环 ----
x_f = x_f + A_LPF * (x - x_f);
if ol_on
    if abs(alpha) < FREEZE && abs(x_f) > DEADBAND
        aref = aref + K_I * x_f * DT;
        if aref >  AREF_MAX, aref =  AREF_MAX; end
        if aref < -AREF_MAX, aref = -AREF_MAX; end
    end
else
    aref = 0.0;
end

% ---- S8 本体（与 9/21 部署逐字相同，只是角度减去 alpha_ref）----
state = [x; v / 5.0; alpha - aref; omega / 10.0];
dlState   = dlarray(single(state), 'CB');
out_actor = extractdata(predict(actorNet, dlState));
mu        = tanh(double(out_actor(1)));
pwm       = mu * MAX_ACTION;

% ---- 可重复的扰动脉冲（每段一次，各组相同）----
extra = 0.0;
if started && seg <= 4 && ts >= KICK_AT && ts < KICK_AT + KICK_DUR
    if seg <= 2
        extra = KICK_AMP;
    else
        extra = -KICK_AMP;
    end
end

% ---- 步进正弦探测（P 段）：8 个频点各 8 s，每段开头 0.5 s 爬升 ----
if started && seg == 5
    tp = ts - 20.0;
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
