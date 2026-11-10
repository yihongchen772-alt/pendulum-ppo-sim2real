#!/usr/bin/env python3
"""11/10 同次对照：按 logs/S17_hw/prediction.md 第 2 节逐条判定，补 s17_hw_analyze.py 没算的几项。

用法：python3 s17_hw_verdict.py   # 读六遍 xlsx 和 analysis_*.json，写 logs/S17_hw/verdict_data.md
补算：切入后 10 s 摆角偏离、零偏遍各控制器全程小车最远、脉冲按方向分组、S17 段 2.4–3.3 Hz 占比分窗口、各项落没落在预测范围。
"""
import json, numpy as np
from s17_hw_analyze import load, start_index, share, DT, T_SEG, W0, W1, CTL

RUNS = ['s17_1', 's17_2', 's17_m20', 's17_p20', 's17_3', 's17_4']
OUT = 'logs/S17_hw'
J = {r['name'].replace('.xlsx', ''): r for r in json.load(open(f'{OUT}/analysis_' + '_'.join(RUNS) + '.json'))}
ZERO = [n for n in RUNS if not J[n]['off']]
L = []; w = L.append

# ---- 第 2、9、10 条：|S| 峰 ----
pk = lambda n, key, p: float(np.nanmax(J[n][key][str(p)])) if str(p) in J[n].get(key, {}) else np.nan
w('# 11/10 同次对照：逐条判定用的数（`s17_hw_verdict.py` 生成）\n\n## 第 2 条 裕度（150 PWM |S| 峰）\n\n|遍|原厂（贴限幅）|S13b|S17|S13b < 原厂|S17 < 原厂|S17 ≤ 5.2|\n|---|---|---|---|---|---|---|\n')
for n in ZERO:
    o, a, b = pk(n, 'S', 0), pk(n, 'S', 1), pk(n, 'S', 2)
    w(f"|{n}|{o:.2f}（{J[n]['S_sat']['0']:.0f}%）|{a:.2f}|{b:.2f}|{'是' if a < o else '否'}|{'是' if b < o else '否'}|{'是' if b <= 5.2 else '否'}|\n")

w('\n## 第 9、10 条 75 PWM\n\n|遍|原厂 75（贴限幅）|×0.9|S13b 75|S17 75|S17 75/150|S13b < 原厂×0.9|S17 < 原厂×0.9|\n|---|---|---|---|---|---|---|---|\n')
for n in ZERO:
    o, a, b = pk(n, 'S75', 0), pk(n, 'S75', 1), pk(n, 'S75', 2)
    w(f"|{n}|{o:.2f}（{J[n]['S75_sat']['0']:.0f}%）|{0.9*o:.2f}|{a:.2f}|{b:.2f}|{b/pk(n,'S',2):.2f}|{'是' if a < 0.9*o else '否'}|{'是' if b < 0.9*o else '否'}|\n")

# ---- 第 3、4 条：用哪几遍定量 ----
w('\n## 第 1、3、4 条 漂移与比值（校零遍）\n\n|遍|原厂漂移内|S13b 漂移内|S17 漂移内|ΔPWM S13b/原厂|ΔPWM S17/原厂|x std S13b/原厂|x std S17/原厂|\n|---|---|---|---|---|---|---|---|\n')
for n in ZERO:
    d = J[n]['drift']; r = J[n]['ratio']
    w(f"|{n}|{'是' if d['0'] else '否'}|{'是' if d['1'] else '否'}|{'是' if d['2'] else '否'}|{r['dp'][0]:.2f}|{r['dp'][1]:.2f}|{r['xs'][0]:.2f}|{r['xs'][1]:.2f}|\n")

# ---- 预测范围对照（校零遍，两段各算一次） ----
PRED = {  # 键：(seg 字段, 预测 [原厂, S13b, S17])
    'ΔPWM RMS': ('dp', [(23.5, 25.4), (14, 17), (16, 21)]),
    'θ std °': ('th', [(1.0, 1.5), (0.85, 1.05), (0.9, 1.15)]),
    '小车 std mm': ('xs', [(12, 15), (17, 21), (11, 16)]),
    'x 段均值 mm': ('x', [(-10, 10), (-30, 30), (-10, 10)]),
}
w('\n## 静置指标对预测范围（校零遍，每个控制器 4 遍 × 2 段）\n\n|指标|控制器|预测|实测范围|在范围内|\n|---|---|---|---|---|\n')
for lab, (k, rng) in PRED.items():
    for c in range(3):
        v = [J[n]['seg'][str(s)][k] for n in ZERO for s in (c, c + 3)]
        inn = sum(rng[c][0] <= q <= rng[c][1] for q in v)
        w(f'|{lab}|{CTL[c]}|{rng[c][0]}–{rng[c][1]}|{min(v):.1f}…{max(v):.1f}|{inn}/{len(v)}|\n')
b5 = {c: [J[n]['bands'][str(s)][1] for n in ZERO for s in (c, c + 3)] for c in range(3)}
for c, rg in zip(range(3), [(5, 8.5), (13, 18), (8, 12)]):
    w(f'|小车 0.1–0.5 Hz mm|{CTL[c]}|{rg[0]}–{rg[1]}|{min(b5[c]):.1f}…{max(b5[c]):.1f}|{sum(rg[0] <= q <= rg[1] for q in b5[c])}/{len(b5[c])}|\n')
for key, lab, rgs in (('S', '|S| 2.5 Hz（150）', [(2.6, 3.0), (4.1, 4.5), (4.4, 4.9)]),):
    for c in range(3):
        v = [J[n][key][str(c)][1] for n in ZERO]
        w(f'|{lab}|{CTL[c]}|{rgs[c][0]}–{rgs[c][1]}|{min(v):.2f}…{max(v):.2f}|{sum(rgs[c][0] <= q <= rgs[c][1] for q in v)}/{len(v)}|\n')

# ---- 第 6 条：脉冲按方向分组 ----
w('\n## 第 6 条 脉冲（校零遍，按方向分组的中位数）\n\n|组|控制器|n|θ 偏离 °|小车位移 mm|恢复 s（0.05 rad）|恢复 s（3σ）|倒|\n|---|---|---|---|---|---|---|---|\n')
P = [p for n in ZERO for p in J[n]['pulses']]
for g, segs, lab in ((0, (0, 1, 2), '+/−/+（1–3 段）'), (1, (3, 4, 5), '−/+/−（4–6 段）')):
    for c in CTL:
        q = [p for p in P if p['ctl'] == c and p['seg'] in segs]
        med = lambda k: np.nanmedian([p[k] for p in q])
        w(f"|{lab}|{c}|{len(q)}|{med('dth'):.1f}|{med('dx'):.0f}|{med('r5'):.2f}|{med('r3'):.2f}|{sum(p['fall'] for p in q)}|\n")
Pall = [p for n in RUNS for p in J[n]['pulses']]
w(f"\n全部六遍脉冲 {len(Pall)} 次，倒 {sum(p['fall'] for p in Pall)} 次。\n")

# ---- 原始记录补算 ----
w('\n## 切入后 10 s 摆角偏离（相对该段 40–85 s 平均读数，预测 ≤ 5°）、2.4–3.3 Hz 占比分窗口、零偏遍全程小车最远\n\n|遍|段|控制器|切入后 10 s 最大偏离 °|占比 10–40 s|占比 40–85 s|\n|---|---|---|---|---|---|\n')
far = {}
for n in RUNS:
    D = load(n + '.xlsx'); al, x = D['al'], D['x']; k0 = start_index(al); tb = D['t'] - D['t'][k0] + DT
    for s in range(6):
        c = s % 3; a0 = J[n]['seg'][str(s)]['b'] / 1000
        m10 = (tb >= s * T_SEG) & (tb < s * T_SEG + 10); mA = (tb >= s * T_SEG + 10) & (tb < s * T_SEG + 40); mB = (tb >= s * T_SEG + W0) & (tb < s * T_SEG + W1)
        dev = np.degrees(np.abs(al[m10] - a0).max()) if c else np.nan
        w(f"|{n}|{s+1}|{CTL[c]}|{'—' if c == 0 else f'{dev:.1f}'}|{share(al[mA])[0]:.0f}%|{share(al[mB])[0]:.0f}%|\n")
    if J[n]['off']:
        # 全程（扶正到 1344 s）每个控制器小车最远：轮流段按段，探测段按段
        segs = [(s * T_SEG, (s + 1) * T_SEG, s % 3) for s in range(6)] + [(900 + p * 74, 974 + p * 74, p % 3) for p in range(6)]
        far[n] = {c: max(1000 * np.abs(x[(tb >= a) & (tb < b)]).max() for a, b, cc in segs if cc == c) for c in range(3)}
if far:
    w('\n|零偏遍|原厂最远 mm（预测 ≤ 230）|S13b（≤ 300）|S17（≤ 260）|\n|---|---|---|---|\n')
    for n, f in far.items(): w(f'|{n}|{f[0]:.0f}|{f[1]:.0f}|{f[2]:.0f}|\n')

# ---- 第 5 条：居中 ----
w('\n## 第 5 条 居中（零偏遍 x 段均值 mm，40–85 s）\n\n|遍|读数 mrad|原厂|S13b|S17|S17 两段 ≤ 20|原厂两段 ≥ 60|\n|---|---|---|---|---|---|---|\n')
for n in RUNS:
    if not J[n]['off']: continue
    s = J[n]['seg']; X = lambda c: (s[str(c)]['x'], s[str(c + 3)]['x'])
    w(f"|{n}|{J[n]['offset']:+.1f}|{X(0)[0]:+.0f} / {X(0)[1]:+.0f}|{X(1)[0]:+.0f} / {X(1)[1]:+.0f}|{X(2)[0]:+.1f} / {X(2)[1]:+.1f}|"
      f"{'是' if max(map(abs, X(2))) <= 20 else '否'}|{'是' if min(map(abs, X(0))) >= 60 else '否'}|\n")
    w(f"|{n}（|S| 峰）||{pk(n,'S',0):.2f}（预测 ≥ 5.7–6.7）|—|{pk(n,'S',2):.2f}（预测 5.0–5.8）|||\n")

# ---- 第 7 条：零偏跳变 ----
w('\n## 第 7 条 零偏（各段读数均值 mrad）\n\n|遍|' + '|'.join(f'段{s+1}' for s in range(6)) + '|极差|\n|---|' + '---|' * 7 + '\n')
for n in RUNS:
    b = [J[n]['seg'][str(s)]['b'] for s in range(6)]
    w(f'|{n}|' + '|'.join(f'{q:+.1f}' for q in b) + f'|{max(b)-min(b):.1f}|\n')

open(f'{OUT}/verdict_data.md', 'w').write(''.join(L)); print(''.join(L))
