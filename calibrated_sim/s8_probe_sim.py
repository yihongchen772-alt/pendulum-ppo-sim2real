#!/usr/bin/env python3
"""S8 闭环探测仿真（s8_bias_probe.m / calib_probe_sim.m 的 Python 移植，2026-09-28）。

与 MATLAB 版逐行对应：非线性小车–摆杆（Euler，4 ms）、执行器 ka·pwm/42 + 一阶滞后 fa、角度纯延迟 tal
（16 格缓存线性插值）、量化、两个二阶微分器（小车 1 Hz、摆杆 0.5 Hz，Tustin）、S8 网络（ReLU，tanh 输出）、
步进正弦探测（每频点 8 s，0.5 s 爬升）、角加速度过程噪声 DIST。

新增（用来做 2–2.5 Hz 缺口的反事实检验）：可在测量端给小车位置、摆角各加一个增益与额外延迟
（corr=dict(gx=增益, dx=延迟秒, ga=增益, da=延迟秒)），让仿真的被控对象在某个频率上与真机实测一致。

用法：python3 s8_probe_sim.py   （先复现 MATLAB 预测，再做缺口检验，结果打印并写入 logs/OL_hw_ol_S8_2/）
"""
import numpy as np, pandas as pd

DT = 0.004
MD = dict(ka=1.60114, fa=3.56836, tal=0.02030, B=30.0, Jp=0.0025, dmp=0.024)      # 联合 3 参校准模型（plant_fit2）
FS = [2.0, 2.5, 3.0, 3.5, 3.96, 4.5, 5.0, 6.0]
W = dict(np.load('actor_net_S8_deploy_w.npz'))      # 可用 set_net() 换成别的网络
Mc, Mp, L, G, DMP = 0.4719, 0.1679, 0.2125, 9.81, 0.024


def td2(f0):
    z, wn, K = 0.9, 2 * np.pi * f0, 2 / DT
    a0 = K ** 2 + 2 * z * wn * K + wn ** 2
    return np.array([wn ** 2 * K, 0, -wn ** 2 * K]) / a0, np.array([1, 2 * (wn ** 2 - K ** 2) / a0, (K ** 2 - 2 * z * wn * K + wn ** 2) / a0])


BO, AO = td2(0.5)
BV, AV = td2(1.0)


def dyn(s, F, B, Jp, dmp=DMP, tc=0.0):
    xd, th, thd = s[1], s[2], s[3]; ml = Mp * L; c = np.cos(th); sn = np.sin(th)
    m11, m12, m22 = Mc + Mp, ml * c, Jp + ml * L
    f1 = F - B * xd + ml * thd ** 2 * sn; f2 = -dmp * thd - tc * np.tanh(thd / 0.01) + Mp * G * L * sn        # 转轴：粘性 + 库仑（平滑）
    det = m11 * m22 - m12 * m12
    return np.array([xd, (m22 * f1 - m12 * f2) / det, thd, (m11 * f2 - m12 * f1) / det])


def set_net(w):
    global W
    W = w


NOTCH_B = np.array([0.0726429761, -0.1447249428, 0.0, 0.1447249428, -0.0726429761])      # 陷波速度估计器（plant_fit.py）
NOTCH_A = np.array([1.0, -3.8298638599, 5.5030952803, -3.5161956822, 0.8429687500])
K_LQR_FACTORY = np.array([-38.7298, 164.2110, -76.3417, 41.0570])      # 原厂 Q=diag([3000,5000,1,1])、R=2
_Rm, _rmp, _Kt, _Kb, _Beq, _Mt = 2.12, 0.013745, 0.08505, 0.11693, 30.0, 0.4719 + 0.1679
_b1, _b0 = _Rm * _rmp ** 2 * _Mt, _Rm * _rmp ** 2 * _Beq + _Kt * _Kb            # F/Volt 分子
_a1, _a0 = _Kt * _rmp * _Mt, _Kt * _rmp * _Beq                                    # F/Volt 分母
_KT = 2 / DT
FV = ((_b1 * _KT + _b0), (_b0 - _b1 * _KT), (_a0 - _a1 * _KT), (_a1 * _KT + _a0))  # Tustin 离散系数


def net(o):
    h = np.maximum(W['W1'] @ o + W['b1'][:, 0], 0); h = np.maximum(W['W2'] @ h + W['b2'][:, 0], 0)
    return np.tanh(W['W3'][0] @ h + W['b3'][0, 0])


def probe_sim(freqs, seed=1, BA=0.0, TW=80.0, amp=150.0, md=MD, dist=2.0, q_th=(360 / 2048) * np.pi / 180, corr=None):
    rng = np.random.default_rng(seed)
    dist = md.get('_dist', dist); q_th = md.get('_q_th', q_th); sn = md.get('_sn', 0.0)     # 可由 md 覆盖：过程噪声、角度量化、角度测量噪声（rad）
    nseq = md.get('_noise_seq'); noff = int(rng.integers(0, len(nseq))) if nseq is not None else 0   # 真机角度噪声序列（循环取用）
    noff = md.get('_noff', noff)                                        # 核对用：固定噪声起点
    corr = corr or {}
    gx, dx, ga, da = corr.get('gx', 1.0), corr.get('dx', 0.0), corr.get('ga', 1.0), corr.get('da', 0.0)
    q_x = 0.658 / 5268
    T = TW + 8 * len(freqs); N = int(round(T / DT))
    s = np.array([0.0, 0.0, (rng.random() - 0.5) * 0.02, 0.0])
    if '_th0' in md: s[2] = md['_th0']                                   # 核对用：固定初始角度
    NB = 64; bth = np.full(NB, s[2]); bx = np.zeros(NB)
    dly_a = (md['tal'] + da) / DT; dly_x = dx / DT
    ua = np.zeros(3); ya = np.zeros(3); ux = np.zeros(3); yx = np.zeros(3)
    Fa = 0.0; tauA = 1 / (2 * np.pi * md['fa'])
    ctrl = md.get('_ctrl', 'net'); fv_u = fv_y = 0.0                     # 'net' = 神经网络；'lqr_factory' = 原厂 LQR 块；'lqr_fcn' = 9/08 代码版 LQR
    scale = md.get('_scale', 1.0); om_mode = md.get('_omega', 'std')     # 网络输出缩放（克隆×0.75）；角速度来源 'std' 或 'notch'
    nb_u = np.zeros(5); nb_y = np.zeros(4)                              # 陷波速度估计器状态
    dly_u = md.get('ta', 0.0) / DT; ub = np.zeros(64)                 # 执行器纯延迟（PWM 历史缓存，线性插值）
    U = np.zeros(N); PR = np.zeros(N); XM = np.zeros(N); AM = np.zeros(N); XT = np.zeros(N)
    for k in range(N):
        tt = (k + 1) * DT
        bth = np.roll(bth, 1); bth[0] = s[2]; bx = np.roll(bx, 1); bx[0] = s[0]
        i0 = int(dly_a); fr = dly_a - i0; thm = (1 - fr) * bth[i0] + fr * bth[i0 + 1]
        j0 = int(dly_x); fx = dly_x - j0; xm = (1 - fx) * bx[j0] + fx * bx[j0 + 1]
        nz = (sn * rng.standard_normal() if sn > 0 else 0.0) + (nseq[(k + noff) % len(nseq)] if nseq is not None else 0.0)
        th_q = np.round((ga * thm + nz) / q_th) * q_th; x_q = np.round(gx * xm / q_x) * q_x; al = -th_q + BA
        ua = np.array([al, ua[0], ua[1]]); ya = np.array([BO @ ua - AO[1] * ya[0] - AO[2] * ya[1], ya[0], ya[1]])
        ux = np.array([x_q, ux[0], ux[1]]); yx = np.array([BV @ ux - AV[1] * yx[0] - AV[2] * yx[1], yx[0], yx[1]])
        if om_mode == 'notch':                                          # 陷波估计器：ω = 0.87·notch(α)（与 plant_fit.loop 一致）
            nb_u = np.r_[al, nb_u[:4]]; yn = (NOTCH_B @ nb_u - NOTCH_A[1:] @ nb_y); nb_y = np.r_[yn, nb_y[:3]]; om_obs = 0.87 * yn
        else:
            om_obs = 2.0 * ya[0]                                        # 部署链路里的 ω（OMEGA_SCALE 使之为真值 2 倍）
        o = np.array([x_q, yx[0] / 5, al, om_obs / 10])
        p = 0.0
        if tt > TW:
            sg = int((tt - TW) // 8)
            if sg < len(freqs):
                ts = tt - TW - sg * 8; p = amp * min(ts / 0.5, 1) * np.sin(2 * np.pi * freqs[sg] * ts)
        if ctrl == 'lqr_factory':
            uF = -(K_LQR_FACTORY @ np.array([x_q, al, yx[0], ya[0]]))
            uF = max(min(uF, 35.0), -35.0)
            fv_y = (FV[0] * uF + FV[1] * fv_u - FV[2] * fv_y) / FV[3]; fv_u = uF
            pl = fv_y * 1000 / 12
            if abs(pl) <= 100: pl *= 1.2
            pl = max(min(pl, 1000), -1000)
            pwm = max(min(pl + p, 1000), -1000)
        elif ctrl == 'lqr_fcn':                                         # 9/08 实验单的 fcn：u=-K[x,α,v,ω(部署链路)]，±35 N，×52.18
            uF = -(md['_K'] @ np.array([x_q, al, yx[0], om_obs]))
            pwm = max(min(max(min(uF, 35.0), -35.0) * 52.18 + p, 1000), -1000)
        else:
            pwm = max(min(scale * net(o) * 1000 + p, 1000), -1000)
        ub = np.roll(ub, 1); ub[0] = pwm; iu = int(dly_u); fu = dly_u - iu; pwm_d = (1 - fu) * ub[iu] + fu * ub[iu + 1]
        Fc = md['ka'] * pwm_d / 42; Fa = Fa + DT / tauA * (Fc - Fa)
        tcp = md.get('tc', 0.0); kf = md.get('kf', 1.0)
        if 'fs' in md:                                           # 实测摩擦模型：库仑（正负向）+ 静摩擦，卡住/滑动两态
            if abs(s[1]) < 2e-3:
                # 假设小车卡住（ẍ = 0）：摆杆方程给出 θ̈，小车方程给出需要的摩擦力
                ml = Mp * L; th, thd = s[2], s[3]
                thdd0 = (Mp * G * L * np.sin(th) - md.get('dmp', DMP) * thd - tcp * np.tanh(thd / 0.01)) / (md['Jp'] + ml * L)
                Freq = Fa + ml * thd ** 2 * np.sin(th) - ml * np.cos(th) * thdd0
                if abs(Freq) <= kf * md['fs']:
                    d = np.array([0.0, 0.0, thd, thdd0]); s[1] = 0.0
                else:
                    Fc = kf * (md['fcp'] if Freq > 0 else md['fcn'])
                    d = dyn(s, Fa - Fc * np.sign(Freq), md['B'], md['Jp'], md.get('dmp', DMP), tcp)
            else:
                Fc = kf * (md['fcp'] if s[1] > 0 else md['fcn'])
                d = dyn(s, Fa - Fc * np.sign(s[1]), md['B'], md['Jp'], md.get('dmp', DMP), tcp)
        else:
            Ff = md.get('fc', 0.0) * np.tanh(s[1] / 0.01)      # 旧的平滑库仑摩擦（0 = 不加）
            d = dyn(s, Fa - Ff, md['B'], md['Jp'], md.get('dmp', DMP), tcp)
        d[3] += dist * rng.standard_normal()                     # 过程噪声（各分支都加）
        v_old = s[1]; s = s + DT * d
        if 'fs' in md and v_old * s[1] < 0: s[1] = 0.0
        U[k], PR[k], XM[k], AM[k], XT[k] = pwm, p, x_q, al, s[0]
        if abs(s[2]) > 0.5 or abs(s[0]) > 0.33:
            return dict(ok=False, tfail=tt)
    t = np.arange(1, N + 1) * DT; out = dict(ok=True, S=[], Gx=[], Ga=[], sat=[], xprobe=1000 * XT[t > TW].mean(), uamp=[], t=t, AM=AM, U=U)
    for i, f in enumerate(freqs):
        m = (t >= TW + 8 * i + 1) & (t < TW + 8 * (i + 1)); e = np.exp(-2j * np.pi * f * t[m])
        cu = ((U[m] - U[m].mean()) * e).sum()
        out['S'].append(abs(cu) / abs((PR[m] * e).sum()))
        out['Gx'].append(((XM[m] - XM[m].mean()) * e).sum() / cu); out['Ga'].append(((AM[m] - AM[m].mean()) * e).sum() / cu)
        out['sat'].append(np.mean(np.abs(U[m]) >= 999)); out['uamp'].append(np.percentile(np.abs(U[m] - PR[m] - (U[m] - PR[m]).mean()), 99))
    return out


def hw_G(fn):
    """真机：各探测频率的 G_x、G_α（与 s8_gap_2Hz.py 同口径）"""
    d = pd.read_excel(fn); g = lambda s: d[[c for c in d.columns if c.endswith(s)][0]].astype(float).values
    t = d[d.columns[0]].astype(float).values; x, al = g('Controller:2'), g('Controller:4')
    u = d[[c for c in d.columns if c.startswith('Saturation2')][0]].astype(float).values
    k0 = int(np.argmax(np.abs(al) < 0.05)); tb = t - t[k0] + DT; res = {}
    for k, f in enumerate(FS):
        ii = np.where((tb >= 500 + 8 * k + 1.0) & (tb < 500 + 8 * (k + 1)))[0]; tk = tb[ii]; e = np.exp(-2j * np.pi * f * tk)
        cu = ((u[ii] - u[ii].mean()) * e).sum()
        res[f] = (((x[ii] - x[ii].mean()) * e).sum() / cu, ((al[ii] - al[ii].mean()) * e).sum() / cu)
    return res


if __name__ == '__main__':
    import sys
    out = []; w = lambda s: (print(s), out.append(s))
    w('## 1. Python 移植复现 MATLAB 预测（联合 3 参，BA = 0，150 PWM）')
    w('MATLAB（prediction_run2.md，居中行）：1.42–1.47 / 2.82–2.88 / 4.98–5.05 / 3.83–4.04 / 2.51–2.65 / 1.75–1.78 / 1.45–1.47 / 1.18–1.19')
    base = {}
    for sd in (1, 2):
        r = probe_sim(FS, seed=sd)
        if not r['ok']: w(f'seed {sd} 倒下 {r["tfail"]:.1f} s'); continue
        base[sd] = r
        w(f'Python seed {sd}：' + ' / '.join(f'{v:.2f}' for v in r['S']) + f'；探测时小车 {r["xprobe"]:+.0f} mm')

    w('\n## 2. 反事实：测量端补上仿真与真机的被控对象差别（逐频率，增益 + 额外延迟），看 |S| 是否升到真机水平')
    gs = {f: (np.mean([base[s]['Gx'][i] for s in base]), np.mean([base[s]['Ga'][i] for s in base])) for i, f in enumerate(FS)}
    HW = {'S8②（居中，+8 mm）': hw_G('ol_S8_2.xlsx'), 'S8①（−121 mm）': hw_G('ol_S8.xlsx')}
    HW_S = {'S8②（居中，+8 mm）': [3.78, 5.70, 4.84, 3.17, 2.25, 1.80, 1.51, 1.33], 'S8①（−121 mm）': [2.39, 5.53, 4.97, 3.82, 2.29, 2.06, 1.74, 1.34]}
    fmt = lambda q: f'{abs(q):.2f}∠{np.degrees(np.angle(q)):+.0f}°'
    for name, hg in HW.items():
        w(f'\n### 以 {name} 的实测 G 为准')
        w('| f Hz | 真机/仿真 G_x | 真机/仿真 G_α | 补偿（x 增益, x 延迟 ms; α 增益, α 延迟 ms） | 仿真 |S| 未补 | 仿真 |S| 补后（3 个种子） | 真机 |S| |')
        w('|---|---|---|---|---|---|---|')
        for i, f in enumerate(FS):
            Rx, Ra = hg[f][0] / gs[f][0], hg[f][1] / gs[f][1]
            ph_x, ph_a = np.angle(Rx), np.angle(Ra)
            dx = max(-ph_x, 0) / (2 * np.pi * f); da = max(-ph_a, 0) / (2 * np.pi * f)
            lead = [n for n, ph in (('x', ph_x), ('α', ph_a)) if ph > np.radians(1)]
            corr = dict(gx=abs(Rx), dx=dx, ga=abs(Ra), da=da)
            S0 = [probe_sim([f], seed=sd, TW=40)['S'][0] for sd in (1, 2, 3)]
            S1 = []
            for sd in (1, 2, 3):
                r = probe_sim([f], seed=sd, TW=40, corr=corr); S1.append(r['S'][0] if r['ok'] else float('nan'))
            note = f'（{"/".join(lead)} 需要超前，延迟补不了，按 0 处理）' if lead else ''
            w(f"| {f:g} | {fmt(Rx)} | {fmt(Ra)} | {abs(Rx):.2f}, {1000*dx:.1f}; {abs(Ra):.2f}, {1000*da:.1f}{note} | {np.mean(S0):.2f} | "
              + ' / '.join(f'{v:.2f}' for v in S1) + f' | {HW_S[name][i]:.2f} |')
    open('logs/OL_hw_ol_S8_2/s8_probe_sim_output.txt', 'w').write('\n'.join(out) + '\n')
