function [pwm, x_out, v_out, theta_out, omega_out, lqr_out] = fcn_LQR_static(cartpos_counts, rodAngle, cartVel_ms, rodVel_rads, lqr_pwm)
%#codegen
% =========================================================================
% 原厂 LQR 干净静置（上机用）。贴进 RL_Controller 的 MATLAB Function 块时，只把第一行的函数名改回 fcn。
% 接口与 fcn_LQR_probe150.m 相同（5 个输入，第 5 个 lqr_pwm 接原厂 LQR 的输出），不用改线。
% 控制完全由原厂 LQR 块负责，本函数原样转发，不加探测、不加脉冲。
% 目的：10/02 只有 18 s 静置且小车在漂，模型 v1 的静置部分对原厂 LQR 准不准没法判（预测 2.5°，实测 1.6°）。
% 扶正后静置至少 150 s，不碰。第 6 路输出 lqr_out = 原厂 LQR 的 PWM。
% =========================================================================
V_SIGN      = -1.0;       % 勿改（只用于记录）
OMEGA_SCALE = 360/2048;

pwm = lqr_pwm;
if pwm >  1000.0, pwm =  1000.0; end
if pwm < -1000.0, pwm = -1000.0; end

x_out = cartpos_counts; v_out = V_SIGN * cartVel_ms; theta_out = rodAngle * (pi / 180);
omega_out = OMEGA_SCALE * rodVel_rads; lqr_out = lqr_pwm;
end
