function E = v1_env_init(P, th0, noff, BA, x0)
% v1 环境初始化（与 s8_probe_sim.probe_sim 一致）。th0 初始摆角（rad，theta 约定），noff 噪声序列起点（0 起），
% BA 角度零偏（rad），x0 初始小车位置（m，可省略为 0）。
if nargin < 5, x0 = 0.0; end
NB = 64;
E.s   = [x0; 0.0; th0; 0.0];          % [x; xd; theta; thetad]
E.bth = th0 * ones(1, NB);            % 角度历史（测量延迟用）
E.bx  = x0 * ones(1, NB);
E.ub  = zeros(1, NB);                 % PWM 历史（执行器纯延迟用）
E.Fa  = 0.0;                          % 执行器输出力（一阶滞后状态）
E.ua = zeros(1, 3); E.ya = zeros(1, 3); E.ux = zeros(1, 3); E.yx = zeros(1, 3);
E.k   = 0;                            % 已走步数
E.noff = noff; E.BA = BA;
E.dly_a = P.tal / P.DT; E.dly_u = P.ta / P.DT; E.tauA = 1 / (2*pi*P.fa);
end
