#!/usr/bin/env python3
"""两个速度是怎么来的：用真机记录核对 Simulink 的测速链路。

平台上没有测速传感器。小车位置来自增量编码器（0.125 mm/格），摆杆角度来自电位器 + 12 位 ADC（4096 格/圈），
树莓派每 4 ms 拿到位置，用二阶滤波微分器 H(s) = wn^2 s / (s^2 + 2 zeta wn s + wn^2) 算速度
（小车 1 Hz、摆杆 0.5 Hz，zeta = 0.9；模型里是连续 Transfer Fcn 块，求解器 ode4、定步长 0.004 s）。
本脚本按 ode4 把同一个滤波器离线套在记录的 x、alpha 上，看能否复现记录的 v、omega（块没有直通，
这一拍输出的是用上一拍读数积分出的状态）；再看 4 ms 内位置变几格（分辨率问题）、真实角速度落在哪个频段、
滤波后的估计还剩多少。

用法：python3 velocity_chain_check.py [s8.xlsx]
列对应：RL_Controller:2 = x(m)，:3 = v(m/s，已乘 V_SIGN)，:4 = alpha(rad)，:5 = omega(已乘 OMEGA_SCALE，真值的 2 倍)。
"真实角速度"没有直接测量，用不同平滑程度的非因果求导给一个范围。
"""
import sys
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, butter, filtfilt, welch

DT = 0.004
QA, QX = 0.001538, 1.25e-4          # 实测最小步长：角度 rad、位置 m


def diff2(sig, fhz, zeta=0.9):
    # 弹簧-阻尼拖小球：y'' = wn^2 (读数 - y) - 2 zeta wn y'，输出小球速度 y'。ode4，读数在一拍内保持不变
    wn = 2*np.pi*fhz; st = np.array([sig[0], 0.0]); out = np.zeros_like(sig)
    f = lambda st, u: np.array([st[1], wn**2*(u - st[0]) - 2*zeta*wn*st[1]])
    for k, u in enumerate(sig):
        out[k] = st[1]                      # 先输出（用的是上一拍为止的读数），再用这一拍的读数往前积分 4 ms
        k1 = f(st, u); k2 = f(st + DT/2*k1, u); k3 = f(st + DT/2*k2, u); k4 = f(st + DT*k3, u)
        st = st + DT/6*(k1 + 2*k2 + 2*k3 + k4)
    return out


fn = sys.argv[1] if len(sys.argv) > 1 else 's8.xlsx'
d = pd.read_excel(fn)
g = lambda s: d[[c for c in d.columns if c.endswith(s)][0]].astype(float).values
t, x, v, al, om = d['time'].astype(float).values, g('Controller:2'), g('Controller:3'), g('Controller:4'), g('Controller:5')
te = t[np.abs(al) < 0.15].max(); m = (t > te - 45) & (t <= te)          # 干净口径：最后 45 s
print(f'{fn}：{t[m][0]:.0f}–{t[m][-1]:.0f} s')

print('\n1) 按 ode4 离线套滤波器能否复现记录的速度')
for nm, est, ref in (('omega 对 H_0.5Hz[alpha]', diff2(al, 0.5)[m], om[m]), ('v 对 H_1Hz[x]', diff2(x, 1.0)[m], v[m])):
    print(f'   {nm}：比值 {np.dot(est, ref)/np.dot(est, est):+.3f}，相关 {np.corrcoef(est, ref)[0, 1]:.4f}')

print('\n2) 4 ms 内位置变几格（直接差分的分辨率：摆杆 %.2f rad/s/格，小车 %.1f mm/s/格）' % (QA/DT, QX/DT*1000))
for nm, sig, q in (('摆杆', al, QA), ('小车', x, QX)):
    n = np.abs(np.round(np.diff(sig[m])/q)).astype(int)
    print(f'   {nm}：0 格 {np.mean(n == 0)*100:.0f}%，1 格 {np.mean(n == 1)*100:.0f}%，2 格 {np.mean(n == 2)*100:.0f}%，≥3 格 {np.mean(n >= 3)*100:.0f}%')
print(f'   摆杆直接 4 ms 差分的 RMS：{np.sqrt(np.mean((np.diff(al)[m[1:]]/DT)**2)):.2f} rad/s')
i0 = np.where(m)[0][4000]; c0 = al[i0 - 1]
print('   逐拍例子：时间 s | 角度（格，相对）| 变化格数 | 直接差分 rad/s | 控制器拿到的角速度 omega/2 rad/s')
for k in range(i0, i0 + 10):
    dc = round((al[k] - al[k - 1])/QA)
    print(f'     {t[k]:.3f} | {round((al[k] - c0)/QA):+4d} | {dc:+3d} | {dc*QA/DT:+.3f} | {om[k]/2:+.4f}')
print(f'   这 1 s 里：直接差分 {np.diff(al[i0-1:i0+250]).min()/DT:+.1f}…{np.diff(al[i0-1:i0+250]).max()/DT:+.1f} rad/s，'
      f'omega/2 {(om[i0:i0+250]/2).min():+.3f}…{(om[i0:i0+250]/2).max():+.3f} rad/s')

print('\n3) 真实角速度在哪个频段（Savitzky-Golay 求导后的功率占比）')
for w in (11, 21):
    f, P = welch(savgol_filter(al, w, 2, deriv=1, delta=DT)[m], fs=1/DT, nperseg=4096)
    sh = [np.trapezoid(P[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)])/np.trapezoid(P, f)*100
          for lo, hi in ((0, 0.5), (0.5, 2), (2, 5), (5, 125))]
    print(f'   SG{w}：<0.5 Hz {sh[0]:.0f}%，0.5–2 Hz {sh[1]:.0f}%，2–5 Hz {sh[2]:.0f}%，>5 Hz {sh[3]:.0f}%')

print('\n4) 控制器拿到的角速度（omega/2 去掉 OMEGA_SCALE 的 2 倍）对真实角速度')
est = om[m]/2
for nm, wt in (('SG11', savgol_filter(al, 11, 2, deriv=1, delta=DT)), ('SG21', savgol_filter(al, 21, 2, deriv=1, delta=DT)),
               ('低通 5 Hz', np.gradient(filtfilt(*butter(4, 5/125), al), DT)),
               ('低通 10 Hz', np.gradient(filtfilt(*butter(4, 10/125), al), DT))):
    wt = wt[m]; rms = np.sqrt(np.mean(wt**2))
    print(f'   {nm:8s}：真实 RMS {rms:.2f} rad/s，估计只有 {np.sqrt(np.mean(est**2))/rms*100:.1f}%，相关 {np.corrcoef(est, wt)[0, 1]:+.2f}')

print('\n5) 滤波器相对理想求导（幅值、滞后）')
for fhz, nm in ((0.5, '摆杆 0.5 Hz'), (1.0, '小车 1 Hz')):
    wn = 2*np.pi*fhz
    for f in (0.95, 2.5, 4.0):
        s = 2j*np.pi*f; h = wn**2/(s**2 + 2*0.9*wn*s + wn**2)
        print(f'   {nm} 在 {f} Hz：{abs(h)*100:.1f}%，滞后 {-np.angle(h, deg=True):.0f}°')
