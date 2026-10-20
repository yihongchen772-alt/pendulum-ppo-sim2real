#!/usr/bin/env python3
"""S8 对 S12：真机记录分析（配合 fcn_S8S12_ABAB.m 的时间表）。判据在拿到数据前写定，见 logs/S12_hw/prediction.md。

用法：
  python3 s12_hw_analyze.py s12_ab_1.xlsx [s12_ab_2.xlsx]     # 真机记录（可同时给两遍）
  python3 s12_hw_analyze.py --sim logs/S12_hw/sim_run.mat     # 用仿真结果自检本脚本

列对应（fcn_S8S12_ABAB 的输出）：RL_Controller:2 = x(m)，:4 = alpha(rad，原始、未减 aref)，
:6 = aref(rad)，Saturation2:1 = 最终 PWM（含脉冲/探测）。
时间表（从第一次 |alpha - aref| < 0.05 起算）：A1 S8 [0,120) B1 S12 [120,240) A2 S8 [240,360) B2 S12 [360,480)
P1 S8 [480,554) P2 S12 [554,628)。静置口径：每段 50–95 s；脉冲在段内 100 s（±500 PWM × 0.1 s）；
探测段静置 10 s 后八个频点各 8 s（去掉每个频点第 1 s）。
"""
import sys, os, numpy as np

DT, T_SEG, T_P, W0, W1, KICK_AT, P_AMP = 0.004, 120.0, 74.0, 50.0, 95.0, 100.0, 150.0
FREQS = [2.0, 2.5, 3.0, 3.5, 3.96, 4.5, 5.0, 6.0]
NAMES = ['A1 S8', 'B1 S12', 'A2 S8', 'B2 S12']


def load(path, sim=False):
    if sim:
        import scipy.io as sio
        R = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['R']
        return dict(t=R.t, x=R.x, al=R.al, aref=R.aref, pwm=R.pwm)
    import pandas as pd
    d = pd.read_excel(path)
    g = lambda s: d[[c for c in d.columns if c.endswith(s)][0]].astype(float).values
    return dict(t=d[d.columns[0]].astype(float).values, x=g('Controller:2'), al=g('Controller:4'),
                aref=g('Controller:6'), pwm=d[[c for c in d.columns if c.startswith('Saturation2')][0]].astype(float).values)


def share(a):
    """2.4–3.3 Hz 功率占比与 1–8 Hz 主峰占比（与真机分析、v1 体检同一口径）。"""
    a0 = a - a.mean(); P = np.abs(np.fft.rfft(a0 * np.hanning(len(a0)))) ** 2; f = np.fft.rfftfreq(len(a0), DT)
    tot = P[f > 0.2].sum(); k = np.argmax(P * ((f > 1) & (f < 8)))
    return 100 * P[(f >= 2.4) & (f <= 3.3)].sum() / tot, 100 * P[max(k - 2, 0):k + 3].sum() / tot, f[k]


def recover(tb, al, t0, a0, band):
    m = (tb >= t0) & (tb < t0 + 10); ok = np.abs(al[m] - a0) <= band; c = 0; need = int(1 / DT)
    for i, o in enumerate(ok):
        c = c + 1 if o else 0
        if c >= need: return tb[m][i - need + 1] - t0
    return np.nan


def analyse(D, name, w):
    t, x, al, aref, pwm = D['t'], D['x'], D['al'], D['aref'], D['pwm']
    k0 = int(np.argmax(np.abs(al - aref) < 0.05)); tb = t - t[k0] + DT
    w(f'\n# {name}\n\n扶正时刻 t = {t[k0]:.2f} s；记录总长 {t[-1]:.1f} s（需 ≥ {t[k0] + 628:.0f} s 才完整）。x 负 = 物理右。\n')
    fell = np.where(np.abs(al) > np.radians(25))[0]
    if len(fell): w(f'**|α| > 25° 的时刻（倒或被扶）：** {np.unique(np.round(tb[fell])).astype(int)[:20]} s\n')

    w('\n## 静置段（每段 50–95 s）\n\n|段|θ std °|PWM std|ΔPWM RMS|x 均值 mm|x std mm|2.4–3.3 Hz 占比|主峰占比（Hz）|aref 末 mrad|\n|---|---|---|---|---|---|---|---|---|\n')
    S = {}
    for s in range(4):
        m = (tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1)
        if m.sum() < 1000: w(f'|{NAMES[s]}|无数据||||||||\n'); continue
        sh, pk, fpk = share(al[m])
        S[s] = r = dict(th=np.degrees(al[m].std()), pwm=pwm[m].std(), dp=np.sqrt(np.mean(np.diff(pwm[m]) ** 2)),
                        x=1000 * x[m].mean(), xs=1000 * x[m].std(), sh=sh)
        w(f"|{NAMES[s]}|{r['th']:.2f}|{r['pwm']:.0f}|{r['dp']:.1f}|{r['x']:+.1f}|{r['xs']:.1f}|{sh:.0f}%|{pk:.0f}%（{fpk:.2f}）|{1000*aref[m][-1]:+.1f}|\n")
    if all(s in S for s in range(4)):
        dth = S[2]['th'] / S[0]['th'] - 1; dpw = S[2]['pwm'] / S[0]['pwm'] - 1
        w(f"\n- 漂移（A1 → A2，S8 自己和自己比）：θ std {100*dth:+.0f}%，PWM std {100*dpw:+.0f}%。"
          f"{'超过 10% ⇒ 本次静置比较只作定性。' if max(abs(dth), abs(dpw)) > 0.10 else '在 10% 内。'}\n")
        for k, lab in (('th', 'θ std'), ('pwm', 'PWM std'), ('dp', 'ΔPWM RMS'), ('xs', 'x std')):
            a = (S[0][k] + S[2][k]) / 2; b = (S[1][k] + S[3][k]) / 2
            w(f"- {lab}：S12 / S8 = {b:.2f} / {a:.2f}（{100*(b/a-1):+.0f}%）\n")

    w('\n## 脉冲（段内 100 s，500 PWM × 0.1 s；A1/B1 为 +，A2/B2 为 −）\n\n|段|θ 最大偏离 °|小车最大位移 mm|恢复 s（≤0.05 rad 持续 1 s）|是否倒|\n|---|---|---|---|---|\n')
    for s in range(4):
        t0 = s * T_SEG + KICK_AT; pre = (tb >= t0 - 5) & (tb < t0); m3 = (tb >= t0) & (tb < t0 + 3); m10 = (tb >= t0) & (tb < t0 + 10)
        if pre.sum() == 0 or m10.sum() < 1000: continue
        a0 = al[pre].mean()
        w(f"|{NAMES[s]}|{np.degrees(np.abs(al[m3] - a0).max()):.1f}|{1000*np.abs(x[m3] - x[pre].mean()).max():.0f}|"
          f"{recover(tb, al, t0, a0, 0.05):.2f}|{'**是**' if np.abs(al[m10]).max() > np.radians(25) else '否'}|\n")

    w('\n## 闭环灵敏度 |S(f)|（150 PWM）\n\n|段|' + '|'.join(f'{f:g}' for f in FREQS) + '|峰值|探测时小车 mm|\n|' + '---|' * (len(FREQS) + 3) + '\n')
    Sv = {}
    for p, lab in ((0, 'P1 S8'), (1, 'P2 S12')):
        p0 = 4 * T_SEG + p * T_P + 10.0; mag = []
        for i, f in enumerate(FREQS):
            m = (tb >= p0 + i * 8 + 1.0) & (tb < p0 + (i + 1) * 8)
            if m.sum() < 1000: mag.append(np.nan); continue
            tk = tb[m] - (p0 + i * 8); pr = P_AMP * np.sin(2 * np.pi * f * tk); e = np.exp(-2j * np.pi * f * tk)
            u = pwm[m] - pwm[m].mean(); mag.append(abs((u * e).sum()) / abs((pr * e).sum()))
        Sv[p] = mag; mx = (tb >= p0) & (tb < p0 + 64)
        xr = f'{1000*x[mx].min():+.0f}…{1000*x[mx].max():+.0f}' if mx.any() else '—'
        w(f'|{lab}|' + '|'.join(f'{v:.2f}' for v in mag) + f'|{np.nanmax(mag) if not np.all(np.isnan(mag)) else np.nan:.2f}|{xr}|\n')

    # ---- 主判据（事前写定）----
    s8, s12 = Sv[0][1], Sv[1][1]
    w(f'\n## 主判据：2.5 Hz 的 |S|\n\n- S8 {s8:.2f}，S12 {s12:.2f}（事前预测：S8 5.5–5.7，S12 3.0–3.5；v1 里 5.1 / 2.9）。\n')
    if np.isnan(s8) or np.isnan(s12): v = '数据不全，不判。'
    elif s12 <= 4.0 and s8 >= 5.0: v = '**成立**：S12 ≤ 4.0 且 S8 ≥ 5.0，在 v1 里得到的裕度带到了真机上。'
    elif s12 >= 5.0: v = '**不成立**：S12 ≥ 5.0，v1 在这个方向不准。'
    else: v = '**不下结论**：落在中间区域。'
    w(f'- 判定：{v}\n')
    return Sv


def main(argv):
    out = []; w = out.append
    if argv[1] == '--sim':
        analyse(load(argv[2], sim=True), os.path.basename(argv[2]) + '（仿真自检）', w); dst = 'logs/S12_hw/analysis_selftest.md'
    else:
        for f in argv[1:]: analyse(load(f), os.path.basename(f), w)
        dst = 'logs/S12_hw/analysis_' + '_'.join(os.path.splitext(os.path.basename(f))[0] for f in argv[1:]) + '.md'
    os.makedirs('logs/S12_hw', exist_ok=True); open(dst, 'w').write(''.join(out)); print(''.join(out)); print(f'写入 {dst}')


if __name__ == '__main__':
    main(sys.argv)
