#!/usr/bin/env python3
"""原厂 LQR 块、S13b、S17 同次轮流对照：真机记录分析（配合 fcn_LQRS13S17.m 的时间表）。判据在拿到数据前写定，见 logs/S17_hw/prediction.md。

用法：
  python3 s17_hw_analyze.py s17_1.xlsx [s17_2.xlsx ...]     # 真机记录，校零遍和零偏遍都可以给，脚本按平均读数自动分开
  python3 s17_hw_analyze.py --sim logs/S17_hw/sim_run.mat   # 用仿真结果自检本脚本

列对应（fcn_LQRS13S17 的输出）：RL_Controller:2 = x(m)，:4 = alpha(rad)，:6 = 平均读数角(rad)，Saturation2:1 = 最终 PWM（含脉冲/探测）。
时间表（从 |alpha| < 0.15 连续 1 s 起算）：1 原厂 2 S13b 3 S17 4 原厂 5 S13b 6 S17 各 150 s；7 原厂 8 S13b 9 S17 各 74 s 探测（150 PWM）；10 原厂 11 S13b 12 S17 同样探测（75 PWM）。
静置口径：1–6 段每段 40–85 s；脉冲在段内 90/110/130 s（±500 PWM × 0.1 s）；探测段静置 10 s 后八个频点各 8 s（去掉每个频点第 1 s）。
零偏遍：第 1 段 40–85 s 的平均读数超过 12 mrad，S13b 段没有脉冲和探测，上机程序和本脚本用同一条规则。
"""
import sys, os, json, numpy as np
from scipy.signal import butter, filtfilt

DT, T_SEG, T_P, W0, W1, N_HOLD, TH_OFF = 0.004, 150.0, 74.0, 40.0, 85.0, 250, 0.012
PROBES = ((0, 150.0, 'S'), (3, 75.0, 'S75'))   # (第几个探测段开始, 幅度 PWM, 结果键)
KICKS = (90.0, 110.0, 130.0)
FREQS = [2.0, 2.5, 3.0, 3.5, 3.96, 4.5, 5.0, 6.0]
CTL = ['原厂', 'S13b', 'S17']
OUT = 'logs/S17_hw'
BANDS = ((0.02, 0.1, '0.02–0.1 Hz'), (0.1, 0.5, '0.1–0.5 Hz（慢晃）'), (0.5, 20.0, '0.5–20 Hz'))


def load(path, sim=False):
    if sim:
        import scipy.io as sio
        R = sio.loadmat(path, squeeze_me=True, struct_as_record=False)['R']
        return dict(t=R.t, x=R.x, al=R.al, am=R.am, pwm=R.pwm)
    import pandas as pd
    d = pd.read_excel(path)
    g = lambda s: d[[c for c in d.columns if c.endswith(s)][0]].astype(float).values
    return dict(t=d[d.columns[0]].astype(float).values, x=g('Controller:2'), al=g('Controller:4'),
                am=g('Controller:6'), pwm=d[[c for c in d.columns if c.startswith('Saturation2')][0]].astype(float).values)


def start_index(al):
    """与 fcn 相同：|alpha| < 0.15 连续 N_HOLD 步，第 N_HOLD 步 t_bal = DT"""
    c = 0
    for k, o in enumerate(np.abs(al) < 0.15):
        c = c + 1 if o else 0
        if c >= N_HOLD: return k
    raise SystemExit('没有找到扶正时刻')


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
    t, x, al, am, pwm = D['t'], D['x'], D['al'], D['am'], D['pwm']
    k0 = start_index(al); tb = t - t[k0] + DT
    m1 = (tb >= W0) & (tb < W1); b1 = al[m1].mean(); off = bool(abs(b1) > TH_OFF)
    w(f'\n# {name}\n\n扶正时刻 t = {t[k0]:.2f} s；记录总长 {t[-1]:.1f} s（需 ≥ {t[k0] + 1344:.0f} s 才完整）。采样间隔中位数 {1000*np.median(np.diff(t)):.2f} ms。x 负 = 物理右。\n')
    w(f"\n这一遍：**{'零偏遍（S13b 段无脉冲、无探测）' if off else '校零遍'}**，第 1 段 40–85 s 平均读数 {1000*b1:+.1f} mrad。\n")
    fell = np.where(np.abs(al[k0:]) > np.radians(25))[0]
    if len(fell): w(f'**|α| > 25° 的时刻（倒或被扶）：** {np.unique(np.round(tb[k0:][fell])).astype(int)[:20]} s\n')

    w('\n## 静置段（每段 40–85 s）\n\n|段|控制器|θ std °|θ 峰 °（去均值）|PWM std|ΔPWM RMS|读数均值（零偏）mrad|x 均值 mm|x std mm|2.4–3.3 Hz 占比|主峰占比（Hz）|\n|---|---|---|---|---|---|---|---|---|---|---|\n')
    S = {}
    for s in range(6):
        m = (tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1)
        if m.sum() < 1000: w(f'|{s+1}|{CTL[s%3]}|无数据|||||||||\n'); continue
        sh, pk, fpk = share(al[m]); a = al[m] - al[m].mean()
        S[s] = r = dict(th=np.degrees(al[m].std()), pk=np.degrees(np.abs(a).max()), pwm=pwm[m].std(), dp=np.sqrt(np.mean(np.diff(pwm[m]) ** 2)),
                        b=1000 * al[m].mean(), x=1000 * x[m].mean(), xs=1000 * x[m].std(), sh=sh)
        w(f"|{s+1}|{CTL[s%3]}|{r['th']:.2f}|{r['pk']:.1f}|{r['pwm']:.0f}|{r['dp']:.1f}|{r['b']:+.1f}|{r['x']:+.1f}|{r['xs']:.1f}|{sh:.0f}%|{pk:.0f}%（{fpk:.2f}）|\n")

    res = dict(name=name, off=off, offset=float(1000 * b1), drift={}, ratio={}, seg={int(k): {q: float(v) for q, v in r.items()} for k, r in S.items()})
    if all(s in S for s in range(6)):
        w('\n## 漂移（同一控制器前后两段，判据 10%）\n\n')
        for c in range(3):
            dth = S[c + 3]['th'] / S[c]['th'] - 1; dpw = S[c + 3]['pwm'] / S[c]['pwm'] - 1
            res['drift'][c] = bool(max(abs(dth), abs(dpw)) <= 0.10)
            w(f"- {CTL[c]}：θ std {100*dth:+.0f}%，PWM std {100*dpw:+.0f}% ⇒ {'在 10% 内' if res['drift'][c] else '超过 10%'}\n")
        w(f"- 这一遍定量比较可用：S13b 对原厂 {'是' if res['drift'][0] and res['drift'][1] else '否'}；S17 对原厂 {'是' if res['drift'][0] and res['drift'][2] else '否'}\n")
        w('\n## 比值（两段平均；只有至少两遍都“漂移在 10% 内、差别超过 10%、方向相同”才写更好/更差）\n\n|量|原厂|S13b|S17|S13b/原厂|S17/原厂|\n|---|---|---|---|---|---|\n')
        for k, lab in (('th', 'θ std °'), ('pk', 'θ 峰 °'), ('pwm', 'PWM std'), ('dp', 'ΔPWM RMS'), ('xs', 'x std mm')):
            v = [(S[c][k] + S[c + 3][k]) / 2 for c in range(3)]
            res['ratio'][k] = (float(v[1] / v[0]), float(v[2] / v[0]))
            w(f"|{lab}|{v[0]:.2f}|{v[1]:.2f}|{v[2]:.2f}|{v[1]/v[0]:.2f}|{v[2]/v[0]:.2f}|\n")
        w('\n### 居中（段均值，mm）\n\n')
        for c in range(3):
            w(f"- {CTL[c]}：x 均值 {S[c]['x']:+.1f} / {S[c+3]['x']:+.1f} mm，读数均值 {S[c]['b']:+.1f} / {S[c+3]['b']:+.1f} mrad\n")
        w('\n### 小车位置分频段 std（mm，二阶带通、零相位）\n\n|段|控制器|' + '|'.join(b[2] for b in BANDS) + '|\n|---|---|---|---|---|\n')
        bd = {}
        for s in range(6):
            m = (tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1); row = []
            for lo, hi, _ in BANDS:
                bb, aa = butter(2, [lo / (0.5 / DT), hi / (0.5 / DT)], 'band'); row.append(1000 * filtfilt(bb, aa, x)[m].std())
            bd[s] = [float(v) for v in row]
            w(f'|{s+1}|{CTL[s%3]}|' + '|'.join(f'{v:.1f}' for v in row) + '|\n')
        res['bands'] = bd

    w('\n## 脉冲（段内 90/110/130 s，500 PWM × 0.1 s；1–3 段 +/−/+，4–6 段 −/+/−）\n\n|段|控制器|第几 s|θ 最大偏离 °|小车最大位移 mm|恢复 s（≤0.05 rad 持续 1 s）|恢复 s（≤3σ 持续 1 s）|是否倒|\n|---|---|---|---|---|---|---|---|\n')
    pulses = []
    for s in range(6):
        if off and s % 3 == 1: continue
        for tk in KICKS:
            t0 = s * T_SEG + tk; pre = (tb >= t0 - 5) & (tb < t0); m3 = (tb >= t0) & (tb < t0 + 3); m10 = (tb >= t0) & (tb < t0 + 10)
            if pre.sum() == 0 or m10.sum() < 1000: continue
            a0 = al[pre].mean(); sg = np.radians(S[s]['th']) if s in S else np.nan
            dth = np.degrees(np.abs(al[m3] - a0).max()); dx = 1000 * np.abs(x[m10] - x[pre].mean()).max()
            r5, r3 = recover(tb, al, t0, a0, 0.05), recover(tb, al, t0, a0, 3 * sg); fall = bool(np.abs(al[m10]).max() > np.radians(25))
            pulses.append(dict(seg=s, ctl=CTL[s % 3], t=tk, dth=float(dth), dx=float(dx), r5=float(r5), r3=float(r3), fall=fall))
            w(f"|{s+1}|{CTL[s%3]}|{tk:.0f}|{dth:.1f}|{dx:.0f}|{r5:.2f}|{r3:.2f}|{'**是**' if fall else '否'}|\n")
    res['pulses'] = pulses

    for shift, amp, key in PROBES:
        w(f'\n## 闭环灵敏度 |S(f)|（{amp:.0f} PWM）\n\n|段|' + '|'.join(f'{f:g}' for f in FREQS) + '|峰值|峰值处贴限幅|探测时小车 mm|\n|' + '---|' * (len(FREQS) + 4) + '\n')
        Sv = {}; Sp = {}
        for p in range(3):
            if off and p == 1: continue
            p0 = 6 * T_SEG + (shift + p) * T_P + 10.0; mag = []; sat = []
            for i, f in enumerate(FREQS):
                m = (tb >= p0 + i * 8 + 1.0) & (tb < p0 + (i + 1) * 8)
                if m.sum() < 1000: mag.append(np.nan); sat.append(np.nan); continue
                tk = tb[m] - (p0 + i * 8); pr = amp * np.sin(2 * np.pi * f * tk); e = np.exp(-2j * np.pi * f * tk)
                u = pwm[m] - pwm[m].mean(); mag.append(abs((u * e).sum()) / abs((pr * e).sum())); sat.append(100 * np.mean(np.abs(pwm[m]) >= 999))
            mx = (tb >= p0) & (tb < p0 + 64)
            if np.all(np.isnan(mag)): w(f'|{CTL[p]}|无数据|\n'); continue
            Sv[p] = [float(v) for v in mag]; ipk = int(np.nanargmax(mag)); Sp[p] = float(sat[ipk])
            xr = f'{1000*x[mx].min():+.0f}…{1000*x[mx].max():+.0f}' if mx.any() else '—'
            w(f'|{CTL[p]}|' + '|'.join(f'{v:.2f}' for v in mag) + f'|{mag[ipk]:.2f} @{FREQS[ipk]:g}|{sat[ipk]:.0f}%|{xr}|\n')
        res[key] = Sv; res[key + '_sat'] = Sp
    return res


def verdict(R, w):
    """多遍合判（事前规则，prediction.md 第 3 节）。校零遍和零偏遍分开算。"""
    Z = [r for r in R if not r['off'] and r['ratio']]
    if len(Z) >= 2:
        w('\n# 校零遍合判（至少两遍）\n\n|量|S13b 对原厂|S17 对原厂|\n|---|---|---|\n')
        for k, lab in (('th', 'θ std'), ('pk', 'θ 峰'), ('pwm', 'PWM std'), ('dp', 'ΔPWM RMS'), ('xs', 'x std')):
            row = []
            for j, c in ((0, 1), (1, 2)):
                G = [r for r in Z if r['drift'][0] and r['drift'][c]]; v = [r['ratio'][k][j] for r in G]
                if len(G) < 2: row.append(f'可用 {len(G)} 遍，定性（比值 ' + '、'.join(f"{r['ratio'][k][j]:.2f}" for r in Z) + '）')
                elif all(q < 0.9 for q in v): row.append(f'小 {100*(1-max(v)):.0f}–{100*(1-min(v)):.0f}%（{len(G)} 遍）')
                elif all(q > 1.1 for q in v): row.append(f'大 {100*(min(v)-1):.0f}–{100*(max(v)-1):.0f}%（{len(G)} 遍）')
                else: row.append('分不出（' + '、'.join(f'{q:.2f}' for q in v) + '）')
            w(f'|{lab}|{row[0]}|{row[1]}|\n')
    if len(R) >= 1:
        w('\n# 各遍 |S| 峰\n\n|遍|类型|读数 mrad|原厂 150|S13b 150|S17 150|原厂 75|S13b 75|S17 75|S17 75/150|原厂 75 峰处贴限幅|\n|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in R:
            pk = lambda k, p: float(np.nanmax(r[k][p])) if p in r.get(k, {}) else np.nan
            c = [f'{pk(k, p):.2f}' if np.isfinite(pk(k, p)) else '—' for k in ('S', 'S75') for p in range(3)]
            q = pk('S75', 2) / pk('S', 2); sat = r.get('S75_sat', {}).get(0, np.nan)
            w(f"|{r['name']}|{'零偏' if r['off'] else '校零'}|{r['offset']:+.1f}|" + '|'.join(c) + f"|{q:.2f}|{sat:.0f}%|\n")


def main(argv):
    out = []; w = out.append
    if argv[1] == '--sim':
        R = [analyse(load(argv[2], sim=True), os.path.basename(argv[2]) + '（仿真自检）', w)]
        dst = f"{OUT}/analysis_selftest_{os.path.splitext(os.path.basename(argv[2]))[0]}.md"
    else:
        R = [analyse(load(f), os.path.basename(f), w) for f in argv[1:]]; verdict(R, w)
        dst = f'{OUT}/analysis_' + '_'.join(os.path.splitext(os.path.basename(f))[0] for f in argv[1:]) + '.md'
    os.makedirs(OUT, exist_ok=True); open(dst, 'w').write(''.join(out)); print(''.join(out)); print(f'写入 {dst}')
    json.dump(R, open(dst.replace('.md', '.json'), 'w'), ensure_ascii=False, indent=0)   # 论文表和图从这里取数


if __name__ == '__main__':
    main(sys.argv)
