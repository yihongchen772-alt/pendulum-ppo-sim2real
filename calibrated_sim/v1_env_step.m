function E = v1_env_step(E, P, pwm, dist_scale, extra_thdd)
% v1 执行器 + 摩擦 + 动力学 + 欧拉一步（与 s8_probe_sim.probe_sim 一致）。
% pwm：本步给出的 PWM（已限幅）。dist_scale：过程噪声倍数（默认 1）。extra_thdd：额外角加速度（冲击用，默认 0）。
if nargin < 4, dist_scale = 1.0; end
if nargin < 5, extra_thdd = 0.0; end
Mc = 0.4719; Mp = 0.1679; L = 0.2125; G = 9.81; ml = Mp * L;
E.ub = [pwm E.ub(1:end-1)];
iu = floor(E.dly_u); fu = E.dly_u - iu;
pwm_d = (1 - fu) * E.ub(iu + 1) + fu * E.ub(iu + 2);
E.Fa = E.Fa + P.DT / E.tauA * (P.ka * pwm_d / 42 - E.Fa);
s = E.s; Fa = E.Fa; kf = P.kf;
if abs(s(2)) < 2e-3
    % 假设小车卡住（xdd = 0）：摆杆方程给出 thdd，小车方程给出需要的摩擦力
    th = s(3); thd = s(4);
    thdd0 = (Mp*G*L*sin(th) - P.dmp*thd - P.tc*tanh(thd/0.01)) / (P.Jp + ml*L);
    Freq = Fa + ml*thd^2*sin(th) - ml*cos(th)*thdd0;
    if abs(Freq) <= kf * P.fs
        d = [0; 0; thd; thdd0]; s(2) = 0.0;
    else
        if Freq > 0, Fc = kf * P.fcp; else, Fc = kf * P.fcn; end
        d = dynv1(s, Fa - Fc * sign(Freq), P);
    end
else
    if s(2) > 0, Fc = kf * P.fcp; else, Fc = kf * P.fcn; end
    d = dynv1(s, Fa - Fc * sign(s(2)), P);
end
d(4) = d(4) + dist_scale * P.DIST * randn() + extra_thdd;
v_old = s(2); s = s + P.DT * d;
if v_old * s(2) < 0, s(2) = 0.0; end
E.s = s;
end

function ds = dynv1(s, F, P)
Mc = 0.4719; Mp = 0.1679; L = 0.2125; G = 9.81; ml = Mp * L;
xd = s(2); th = s(3); thd = s(4); c = cos(th); sn = sin(th);
m11 = Mc + Mp; m12 = ml * c; m22 = P.Jp + ml * L;
f1 = F - P.B * xd + ml * thd^2 * sn;
f2 = -P.dmp * thd - P.tc * tanh(thd / 0.01) + Mp * G * L * sn;
det = m11 * m22 - m12 * m12;
ds = [xd; (m22*f1 - m12*f2) / det; thd; (m11*f2 - m12*f1) / det];
end
