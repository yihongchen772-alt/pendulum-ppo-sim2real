function [o, E, al, x_q] = v1_env_observe(E, P)
% v1 测量链（与 s8_probe_sim.probe_sim 每步开头一致）：角度延迟 -> 加真机噪声 -> 量化 -> 加零偏 -> 二阶微分器。
% 返回网络观测 o = [x; v/5; alpha; omega/10]（omega 为真值 2 倍，与部署一致）。
E.k = E.k + 1;
E.bth = [E.s(3) E.bth(1:end-1)];  E.bx = [E.s(1) E.bx(1:end-1)];
i0 = floor(E.dly_a); fr = E.dly_a - i0;
thm = (1 - fr) * E.bth(i0 + 1) + fr * E.bth(i0 + 2);
xm  = E.bx(1);
nz  = P.noise(mod(E.k - 1 + E.noff, numel(P.noise)) + 1);
th_q = round((thm + nz) / P.Q_TH) * P.Q_TH;
x_q  = round(xm / P.Q_X) * P.Q_X;
al   = -th_q + E.BA;
E.ua = [al E.ua(1:2)];
E.ya = [P.BO(1)*E.ua(1) + P.BO(2)*E.ua(2) + P.BO(3)*E.ua(3) - P.AO(2)*E.ya(1) - P.AO(3)*E.ya(2), E.ya(1:2)];
E.ux = [x_q E.ux(1:2)];
E.yx = [P.BV(1)*E.ux(1) + P.BV(2)*E.ux(2) + P.BV(3)*E.ux(3) - P.AV(2)*E.yx(1) - P.AV(3)*E.yx(2), E.yx(1:2)];
o = [x_q; E.yx(1) / 5; al; P.OM * E.ya(1) / 10];
end
