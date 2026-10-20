function P = v1_params()
% 校准仿真模型 v1 的参数（2026-10-03 定版，与 sim_model_v1.py 的 V1 逐项相同；说明见 logs/model_v1/model_card.md）。
% 真机角度噪声序列从 v1_assets.mat 读（由 export_v1_assets.py 生成，来自 angle_noise_hw.npz）。
P.DT   = 0.004;
P.Jp   = 0.00238;   P.dmp = 0.0006;  P.tc = 0.00095;                 % 摆杆：转动惯量、转轴粘性、转轴库仑
P.B    = 34.04206067000318;                                          % 小车粘性阻尼
P.fcp  = 1.4445244283469572; P.fcn = 1.532274630479637; P.fs = 2.0;  % 小车库仑（正/负）、静摩擦
P.kf   = 0.8295060174330753;                                         % 摩擦在平衡工况下的系数
P.ka   = 1.6469698769364034; P.fa = 4.569647799893637;               % 执行器增益（推力 = ka*pwm/42）、一阶滞后 Hz
P.ta   = 0.005848074804216543;                                       % 执行器纯延迟 s
P.tal  = 0.019911386105418448;                                       % 角度测量延迟 s
P.DIST = 4.076189447702792;                                          % 过程噪声（加在角加速度上）
P.Q_TH = 1.538e-3;  P.Q_X = 0.658/5268;                              % 量化
P.OM   = 2.0;                                                        % 部署链路里 ω 为真值 2 倍
A = load('v1_assets.mat', 'noise'); P.noise = A.noise(:)';
[P.BO, P.AO] = td2_coef(0.5, P.DT);  [P.BV, P.AV] = td2_coef(1.0, P.DT);
end

function [b, a] = td2_coef(f0, DT)
z = 0.9; wn = 2*pi*f0; K = 2/DT; a0 = K^2 + 2*z*wn*K + wn^2;
b = [wn^2*K 0 -wn^2*K]/a0; a = [1 2*(wn^2 - K^2)/a0 (K^2 - 2*z*wn*K + wn^2)/a0];
end
