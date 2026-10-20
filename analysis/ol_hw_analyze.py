#!/usr/bin/env python3
"""S8 + 慢速外环居中：真机记录分析（配合 fcn_S8_OL.m 的时间表）。

用法：
  python3 ol_hw_analyze.py ol_S8.xlsx            # 真机记录
  python3 ol_hw_analyze.py --sim ol_sim_full_FC0.mat   # 用仿真结果自检本脚本

列对应（fcn_S8_OL 的输出）：RL_Controller:2 = x(m)，:4 = alpha(rad，原始、未减 aref)，
:6 = aref(rad)，Saturation2:1 = 最终 PWM（含脉冲/探测）。

时间表（从第一次 |alpha|<0.05 起算）：A1[0,120) B1[120,240) A2[240,360) B2[360,480) P[480,570)。
静置口径：每段 50–95 s（45 s，外环已收敛、脉冲之前）。脉冲在段内 100 s。
事前判定（实验单写死）：B1、B2 的 |x 均值| < 10 mm，且 θ std、PWM std 不比 A 段均值差 10% 以上；
A1→A2 的 θ std、PWM std 变化 > 10% 时，当次比较降级为定性。
"""
import sys, os, numpy as np

DT, T_SEG, W0, W1, KICK_AT, P_AMP = 0.004, 120.0, 50.0, 95.0, 100.0, 150.0   # P_AMP 与 fcn_S8_OL.m 一致
KICK_AMP = {'ol_S8_2': 500.0}   # 脉冲幅度按记录区分：ol_S8 用 fcn_S8_OL（700），ol_S8_2 用 fcn_S8_OLc（500，P 段外环开）
FREQS = [2.0, 2.5, 3.0, 3.5, 3.96, 4.5, 5.0, 6.0]
NAMES = ['A1', 'B1', 'A2', 'B2']


def load(argv):
    if argv[1] == '--sim':
        import scipy.io as sio
        R = sio.loadmat(argv[2], squeeze_me=True, struct_as_record=False)['R']
        return dict(t=R.t, x=R.x, al=R.al, aref=R.aref, pwm=R.pwm), argv[2]
    import pandas as pd
    d = pd.read_excel(argv[1])
    g = lambda s: d[[c for c in d.columns if c.endswith(s)][0]].astype(float).values
    return dict(t=d[d.columns[0]].astype(float).values, x=g('Controller:2'), al=g('Controller:4'),
                aref=g('Controller:6'), pwm=d[[c for c in d.columns if c.startswith('Saturation2')][0]].astype(float).values), argv[1]


def spec_peak(a):
    a0 = a - a.mean(); F = np.fft.rfft(a0 * np.hanning(len(a0))); P = np.abs(F) ** 2; fr = np.fft.rfftfreq(len(a0), DT)
    k = np.argmax(P * ((fr > 1) & (fr < 8)))
    return fr[k], P[max(k - 2, 0):k + 3].sum() / P[fr > 0.2].sum()


def lpf(x, fc):
    a = 1 - np.exp(-2 * np.pi * fc * DT); y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1])
    return y


def main(argv):
    D, src = load(argv)
    t, x, al, aref, pwm = D['t'], D['x'], D['al'], D['aref'], D['pwm']
    k0 = int(np.argmax(np.abs(al) < 0.05)); tb = t - t[k0] + DT           # 与 fcn 的 t_bal 对齐
    out = []; w = out.append
    w(f'# S8 + 慢速外环居中：分析（{os.path.basename(src)}）\n')
    w(f'扶正时刻 t={t[k0]:.2f} s；记录总长 {t[-1]:.1f} s（需 ≥ {t[k0] + 570:.0f} s 才完整）。x 负 = 物理右。\n')

    # ---- 对齐自检：A 段 aref 应恒为 0，B 段 50 s 后应非零 ----
    chk = []
    for s in range(4):
        m = (tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1)
        if m.sum() == 0: chk.append(f'{NAMES[s]}: 无数据'); continue
        chk.append(f'{NAMES[s]}: |aref| 最大 {np.abs(aref[m]).max()*1000:.2f} mrad')
    w('对齐自检（A 段应为 0，B 段应非零）：' + '；'.join(chk) + '\n')

    # ---- 静置段 ----
    w('\n## 静置段（每段 50–95 s）\n')
    w('|段|外环|x 均值 mm|x std|θ std °|θ 峰 °|PWM std|ΔPWM RMS|θ 谱峰 Hz|峰占比|aref 末 mrad|慢漂 std mm|\n|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    S = {}
    xl5 = lpf(x, 0.5)
    for s in range(4):
        m = (tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1)
        if m.sum() < 1000: continue
        fpk, sh = spec_peak(al[m])
        S[NAMES[s]] = r = dict(x=1000 * x[m].mean(), xs=1000 * x[m].std(), th=np.degrees(al[m].std()), pk=np.degrees(np.abs(al[m] - al[m].mean()).max()),
                               pwm=pwm[m].std(), dp=np.sqrt(np.mean(np.diff(pwm[m]) ** 2)), fpk=fpk, sh=100 * sh, aref=1000 * aref[m][-1], slow=1000 * xl5[m].std())
        w(f"|{NAMES[s]}|{'开' if s in (1, 3) else '关'}|{r['x']:+.1f}|{r['xs']:.1f}|{r['th']:.2f}|{r['pk']:.1f}|{r['pwm']:.0f}|{r['dp']:.1f}|{r['fpk']:.2f}|{r['sh']:.0f}%|{r['aref']:+.2f}|{r['slow']:.1f}|\n")

    # ---- 收敛 ----
    w('\n## 外环收敛\n')
    for s in (1, 3):
        m = (tb >= s * T_SEG) & (tb < s * T_SEG + W1)
        if m.sum() == 0: continue
        fin = aref[(tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1)].mean()
        i = np.argmax(np.abs(aref[m] - fin) < 0.1 * abs(fin)) if abs(fin) > 1e-4 else 0
        lim = np.abs(aref[m]).max() >= 0.0249
        w(f"- {NAMES[s]}：aref 终值 {fin*1000:+.2f} mrad，到 90% 用 {tb[m][i] - s*T_SEG:.1f} s；{'**碰到限幅 ±25 mrad**' if lim else '未碰限幅'}\n")
    if 'A1' in S:
        m = (tb >= 2 * T_SEG) & (tb < 2 * T_SEG + W1); xl = lpf(x, 0.2)
        j = np.argmax(np.abs(xl[m] - S['A1']['x'] / 1000) < 0.005) if m.any() else 0
        w(f"- A2：外环清零后回到 A1 平衡点（±5 mm）用 {tb[m][j] - 2*T_SEG:.1f} s\n")

    # ---- 脉冲 ----
    kick = KICK_AMP.get(os.path.splitext(os.path.basename(src))[0], 700.0)
    w(f'\n## 扰动脉冲（段内 100 s，{kick:.0f} PWM × 0.1 s；A1/B1 为 +，A2/B2 为 −）\n')
    w('|段|θ 最大偏离 °|小车最大位移 mm|恢复 s（3 倍噪声带）|恢复 s（≤2.9° 持续 1 s）|是否倒|\n|---|---|---|---|---|---|\n')
    for s in range(4):
        t0 = s * T_SEG + KICK_AT
        pre = (tb >= t0 - 5) & (tb < t0); m3 = (tb >= t0) & (tb < t0 + 3); m = (tb >= t0) & (tb < t0 + 10)
        if pre.sum() == 0 or m.sum() == 0: continue
        dth = np.degrees(np.abs(al[m3] - al[pre].mean()).max()); dx = 1000 * np.abs(x[m3] - x[pre].mean()).max()
        fell = np.abs(al[m]).max() > np.radians(25)
        recs = []
        for band in (max(3 * al[pre].std(), 0.01), 0.05):
            ok = np.abs(al[m] - al[pre].mean()) <= band; c = 0; rec = np.nan; need = int(1 / DT)
            for i, o in enumerate(ok):
                c = c + 1 if o else 0
                if c >= need: rec = tb[m][i - need + 1] - t0; break
            recs.append(rec)
        w(f"|{NAMES[s]}|{dth:.1f}|{dx:.0f}|{recs[0]:.2f}|{recs[1]:.2f}|{'**是**' if fell else '否'}|\n")

    # ---- 闭环灵敏度 ----
    w(f'\n## S8 闭环灵敏度 |S(f)|（P 段，步进正弦 {P_AMP:.0f} PWM）\n')
    t0 = 4 * T_SEG + 20.0; mag = []; sat = []
    for i, f in enumerate(FREQS):
        m = (tb >= t0 + i * 8 + 1.0) & (tb < t0 + (i + 1) * 8)
        if m.sum() < 1000: mag.append(np.nan); continue
        tk = tb[m] - (t0 + i * 8); p = P_AMP * np.sin(2 * np.pi * f * tk); e = np.exp(-2j * np.pi * f * tk)
        mag.append(abs((pwm[m] * e).sum()) / abs((p * e).sum())); sat.append(np.mean(np.abs(pwm[m]) >= 999))
    if not np.all(np.isnan(mag)):
        w('|f Hz|' + '|'.join(f'{f:g}' for f in FREQS) + '|\n|' + '---|' * (len(FREQS) + 1) + '\n')
        w('|S8|' + '|'.join('—' if np.isnan(v) else f'{v:.2f}' for v in mag) + '|\n')
        w('|贴限幅|' + '|'.join(f'{100*q:.0f}%' for q in sat) + '|\n')
        xm = [1000 * x[(tb >= t0 + i * 8 + 1.0) & (tb < t0 + (i + 1) * 8)].mean() for i in range(len(FREQS))]
        w('|小车均值 mm|' + '|'.join(f'{v:+.0f}' for v in xm) + '|\n')
        mp = (tb >= t0) & (tb < t0 + 64)
        w(f'\n探测期间小车均值 {1000*x[mp].mean():+.1f} mm；aref 均值 {1000*aref[mp].mean():+.2f} mrad（0 = P 段外环关）。\n')
        if max(sat) > 0.05:
            w('\n**有频点 PWM 贴限幅超过 5%：那些频点的 |S| 被削低，只能当下限。**\n')
        w(f'\nS8 的 Ms（8 点最大）= {np.nanmax(mag):.2f} @ {FREQS[int(np.nanargmax(mag))]:g} Hz；对照（9/07–9/08，200 PWM）：未饱和的只有克隆×0.75 4.2 与克隆+陷波 4.0；克隆×1 5.5、v25b 5.3、LQR-A/C 5.1/5.3 在峰值处 34–47% 时间贴限幅，是下限。\n')
    else:
        w('（记录不够长，没有 P 段数据）\n')

    # ---- 判定 ----
    w('\n## 事前判定\n')
    if all(k in S for k in NAMES):
        A = {q: (S['A1'][q] + S['A2'][q]) / 2 for q in ('th', 'pwm')}
        drift = max(abs(S['A2'][q] - S['A1'][q]) / S['A1'][q] for q in ('th', 'pwm'))
        c1 = all(abs(S[b]['x']) < 10 for b in ('B1', 'B2'))
        c2 = all(S[b][q] <= 1.10 * A[q] for b in ('B1', 'B2') for q in ('th', 'pwm'))
        w(f"- 居中：B1 {S['B1']['x']:+.1f} mm、B2 {S['B2']['x']:+.1f} mm → {'**达标**（<10 mm）' if c1 else '**未达标**'}\n")
        w(f"- 代价：B 的 θ std {S['B1']['th']:.2f}/{S['B2']['th']:.2f}° vs A 均值 {A['th']:.2f}°；PWM std {S['B1']['pwm']:.0f}/{S['B2']['pwm']:.0f} vs {A['pwm']:.0f} → {'**不劣于 A 的 10%**' if c2 else '**比 A 差超过 10%**'}\n")
        w(f"- 当天漂移：A1→A2 最大变化 {100*drift:.0f}% → {'定量比较有效' if drift <= 0.10 else '**超过 10%，本次比较降级为定性**'}\n")
    else:
        w('四段不全，无法判定。\n')

    txt = ''.join(out); print(txt)
    if argv[1] != '--sim':
        od = os.path.join('logs', 'OL_hw_' + os.path.splitext(os.path.basename(src))[0]); os.makedirs(od, exist_ok=True)
        open(os.path.join(od, 'review_ol.md'), 'w').write(txt); print(f'\n写入 {od}/review_ol.md')


if __name__ == '__main__':
    main(sys.argv)
