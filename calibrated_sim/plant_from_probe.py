#!/usr/bin/env python3
"""从 7 次步进正弦探测里，直接算出真机"被控对象"的频率响应（幅值 + 相位），与仿真模型逐频比较。

原理：探测信号 p 从外部叠加在 PWM 上，与传感器噪声、反馈都无关。被控对象关系 θ = G_α·u、x = G_x·u
不管有没有反馈都成立，所以在探测频率上
    G_α(f) = Α(f)/U(f)，  G_x(f) = X(f)/U(f)
（U 是总 PWM，含控制器输出与探测；Α、X 是 fcn 的角度、位置输入）。不需要知道控制器。
另外：灵敏度 S = U/P；两个速度估计器的实际响应 = V/X、Ω/Α。

这些量里包含了从 fcn 输出 PWM 到 fcn 读到 α、x 之间的一切：电机、驱动、机械、传感、延迟。
与仿真的 G_model 相除，差出来的幅值与相位就是仿真缺的东西。

数据：probe2（克隆×1）、v25bprobe、cloneprobe*0.75、ksel1/2/3（LQR A/B/C）、probe_notch（克隆+陷波）。
探测时间表：模型启动后 20 s 起，每频点 8 s（前 0.5 s 爬升），8 个频点。
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

DT = 0.004
FREQS = [2.0, 2.5, 3.0, 3.5, 3.96, 4.5, 5.0, 6.0]
RUNS = [('克隆×1', 'probe2.xlsx', True), ('克隆×0.75', 'cloneprobe*0.75.xlsx', True), ('v25b', 'v25bprobe.xlsx', True),
        ('LQR-A', 'ksel1.xlsx', True), ('LQR-B', 'ksel2.xlsx', True), ('LQR-C', 'ksel3.xlsx', True), ('克隆+陷波', 'probe_notch.xlsx', False)]

# ---- 仿真模型（与训练脚本 dyn 相同，在直立点线性化）----
Mc, Mp, l, g, dmp, Jp, P2F, BEQ = 0.4719, 0.1679, 0.2125, 9.81, 0.024, 0.0025, 42.0, 30.0
ml, J = Mp * l, Jp + Mp * l ** 2


def G_model(f, jp=Jp, p2f=P2F, beq=BEQ, tau=0.0):
    s = 2j * np.pi * f
    A = np.array([[(Mc + Mp) * s ** 2 + beq * s, ml * s ** 2], [ml * s ** 2, J * s ** 2 + dmp * s - Mp * g * l]]) if jp == Jp else \
        np.array([[(Mc + Mp) * s ** 2 + beq * s, ml * s ** 2], [ml * s ** 2, (jp + Mp * l ** 2) * s ** 2 + dmp * s - Mp * g * l]])
    X, TH = np.linalg.solve(A, np.array([1.0 / p2f, 0.0]))
    zoh = np.exp(-s * DT / 2) * np.exp(-s * tau)          # 零阶保持半拍 + 附加纯延迟
    return X * zoh, -TH * zoh                               # alpha = -theta（与仿真/部署约定一致）


def td2_resp(f0, f):
    """仿真里的二阶微分器（Tustin，250 Hz）在频率 f 的响应"""
    z, wn, K = 0.9, 2 * np.pi * f0, 2 / DT
    a0 = K ** 2 + 2 * z * wn * K + wn ** 2
    b = np.array([wn ** 2 * K, 0, -wn ** 2 * K]) / a0
    a = np.array([1, 2 * (wn ** 2 - K ** 2) / a0, (K ** 2 - 2 * z * wn * K + wn ** 2) / a0])
    zi = np.exp(-2j * np.pi * f * DT)
    return (b[0] + b[1] * zi + b[2] * zi ** 2) / (a[0] + a[1] * zi + a[2] * zi ** 2)


def coef(y, t, f):
    y = y - y.mean(); e = np.exp(-2j * np.pi * f * t)
    return (y * e).sum() * 2 / len(y)


def analyze(fn, has_v):
    d = pd.read_excel(fn)
    gc = lambda s: d[[c for c in d.columns if c.endswith(s)][0]].astype(float).values
    t = d[d.columns[0]].astype(float).values
    x, v, al, om, p = gc('Controller:2'), gc('Controller:3'), gc('Controller:4'), gc('Controller:5'), gc('Controller:6')
    u = d[[c for c in d.columns if c.startswith('Saturation2')][0]].astype(float).values
    out = []
    for k, f in enumerate(FREQS):
        m = (t >= 20 + 8 * k + 1.0) & (t < 20 + 8 * (k + 1))
        ok = m.sum() > 1000 and np.abs(al[m]).max() < 0.35 and np.abs(p[m]).max() > 150
        if not ok:
            out.append(None); continue
        tm = t[m]; C = {n: coef(y[m], tm, f) for n, y in dict(p=p, u=u, x=x, al=al, v=v, om=om).items()}
        # 质量：把块分成两半各算一次 G_α，差异小说明估计稳定
        h = m.nonzero()[0]; h1, h2 = h[:len(h) // 2], h[len(h) // 2:]
        ga1 = coef(al[h1], t[h1], f) / coef(u[h1], t[h1], f); ga2 = coef(al[h2], t[h2], f) / coef(u[h2], t[h2], f)
        out.append(dict(S=C['u'] / C['p'], Ga=C['al'] / C['u'], Gx=C['x'] / C['u'],
                        Hv=(C['v'] / C['x']) if has_v else np.nan, Hw=C['om'] / C['al'], rep=abs(ga1 - ga2) / abs((ga1 + ga2) / 2)))
    return out


def deg(z):
    return np.degrees(np.angle(z))


def main():
    R = {}
    for name, fn, hv in RUNS:
        try:
            R[name] = analyze(fn, hv)
        except FileNotFoundError:
            print('缺文件', fn)
    print('\n①  灵敏度 |S|（与 9/07、9/08 报的数应一致）')
    print(f"{'f Hz':<10}" + ''.join(f'{n:>11}' for n in R))
    for k, f in enumerate(FREQS):
        print(f'{f:<10}' + ''.join(f"{abs(R[n][k]['S']):11.2f}" if R[n][k] else f"{'—':>11}" for n in R))

    print('\n②  真机 G_α ÷ 模型 G_α（幅值比 / 相位差°）—— 7 次独立估计；"半块差"= 同一块前后两半的相对差')
    print(f"{'f Hz':<8}" + ''.join(f'{n:>17}' for n in R) + f"{'中位数':>17}")
    med_a = []
    for k, f in enumerate(FREQS):
        gm = G_model(f)[1]; row = []; cells = []
        for n in R:
            r = R[n][k]
            if r is None: cells.append(f"{'—':>17}"); continue
            q = r['Ga'] / gm; row.append(q); cells.append(f"{abs(q):7.2f}/{deg(q):+6.0f}°  ")
        mq = np.median(np.abs(row)) * np.exp(1j * np.median(np.angle(row))); med_a.append(mq)
        print(f'{f:<8}' + ''.join(cells) + f"{abs(mq):7.2f}/{deg(mq):+6.0f}°")
    print('    半块差（G_α 估计的稳定性）：' + '，'.join(f"{n} {np.nanmedian([r['rep'] for r in R[n] if r]):.0%}" for n in R))

    print('\n③  真机 G_x ÷ 模型 G_x（幅值比 / 相位差°）')
    print(f"{'f Hz':<8}" + ''.join(f'{n:>17}' for n in R) + f"{'中位数':>17}")
    med_x = []
    for k, f in enumerate(FREQS):
        gm = G_model(f)[0]; row = []; cells = []
        for n in R:
            r = R[n][k]
            if r is None: cells.append(f"{'—':>17}"); continue
            q = r['Gx'] / gm; row.append(q); cells.append(f"{abs(q):7.2f}/{deg(q):+6.0f}°  ")
        mq = np.median(np.abs(row)) * np.exp(1j * np.median(np.angle(row))); med_x.append(mq)
        print(f'{f:<8}' + ''.join(cells) + f"{abs(mq):7.2f}/{deg(mq):+6.0f}°")

    print('\n④  速度估计器：实测 Ω/Α ÷ 模型(2×0.5 Hz 微分器)，实测 V/X ÷ 模型(1 Hz 微分器)（中位数，幅值比/相位差°）')
    for k, f in enumerate(FREQS):
        hw = [R[n][k]['Hw'] for n in R if R[n][k] and n != '克隆+陷波']
        hv = [R[n][k]['Hv'] for n in R if R[n][k] and not np.isnan(R[n][k]['Hv'])]
        qw = np.median(np.abs(hw)) / abs(2 * td2_resp(0.5, f)); pw = np.median(np.angle(hw)) - np.angle(2 * td2_resp(0.5, f))
        qv = np.median(np.abs(hv)) / abs(td2_resp(1.0, f)); pv = np.median(np.angle(hv)) - np.angle(td2_resp(1.0, f))
        print(f'  {f:<6} Ω/Α: {qw:5.2f} / {np.degrees(pw):+6.1f}°     V/X: {qv:5.2f} / {np.degrees(pv):+6.1f}°')

    # ---- 若差的是纯延迟：相位差对频率做过原点直线拟合 ----
    fa = np.array(FREQS); pa = np.unwrap(np.angle(np.array(med_a))); px = np.unwrap(np.angle(np.array(med_x)))
    for nm, ph in (('G_α', pa), ('G_x', px)):
        tau = -(ph @ (2 * np.pi * fa)) / ((2 * np.pi * fa) @ (2 * np.pi * fa))
        res = np.degrees(ph + 2 * np.pi * fa * tau)
        print(f'\n{nm} 相位差若全算作纯延迟：τ = {tau*1000:.1f} ms；拟合残差 {np.round(res, 0)}°')


if __name__ == '__main__':
    main()
