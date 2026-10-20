#!/usr/bin/env python3
"""
奖励函数的有效性验证 —— 拿真机数据反算奖励，看它排的名次对不对

【为什么做这个】
奖励函数是在仿真里用的，但它存在的全部意义是「高奖励 <=> 真机上好」。
这个对应关系从未验证过。若它在真机数据上排不出正确名次，那它就是坏的，
再讨论加不加速度项都没意义。

【方法】
对每条真机记录的稳态段，用【真机实测的 theta / x / pwm】逐步算奖励：
    r = R_SURV - W_th*(th/0.2)^2 - W_x*(x/0.2)^2 - W_F*(|pwm|/1000)^2
        - W_S*(dpwm/2000)^2*4                      (满难度 d=1)
再和「真机实际好坏」的排名比对。

【真机好坏的基准】用 theta std（抖动），这是项目一贯的首要指标。
另给一个复合基准：theta std 与 PWM std 的等权秩平均。

【然后】加入假想的速度项，看排名是否改善 —— 这就是「该不该加」的直接证据。
"""
import pandas as pd, numpy as np, os
from scipy import stats

DT = 0.004
PROBE_END = {'ksel1.xlsx': 90, 'ksel2.xlsx': 90, 'ksel3.xlsx': 90}

# 与 train_rl_scratch_S11.m 完全一致
R_SURV, W_TH, W_X, W_F, W_S = 10.0, 5.0, 8.0, 1.5, 6.0
TH_LIM, X_LIM, MAXA = 0.2, 0.2, 1000.0

FILES = [
    ('S8',        's8.xlsx'),
    ('克隆x0.85',  'test结果（clone）.xlsx'),
    ('克隆x0.75',  'test结果75（clone）.xlsx'),
    ('LQR-C',     'ksel3.xlsx'),
    ('v25b',      'v25（clone）.xlsx'),
    ('克隆x1.00',  'test6.xlsx'),
    ('混合AC',     'blendac.xlsx'),
    ('LQR-A',     'ksel1.xlsx'),
    ('S11',       's11.xlsx'),
    ('v24',       'v24结果（clone）.xlsx'),
    ('LQR-B',     'ksel2.xlsx'),
    ('S10',       's10.xlsx'),
]

def tail(f, T=45.0):
    d = pd.read_excel(f)
    t = d[d.columns[0]].values
    g = lambda s: [k for k in d.columns if k.endswith(s)][0]
    th = d[g('Controller:4')].astype(float).values
    x  = d[g('Controller:2')].astype(float).values
    v  = d[g('Controller:3')].astype(float).values
    om = d[g('Controller:5')].astype(float).values
    pw = d[[k for k in d.columns if k.startswith('Saturation2')][0]].astype(float).values
    t0 = PROBE_END.get(f, 0)
    m = (t > t0) & (np.abs(th) < 0.15)
    te = t[m].max()
    s = (t > max(t0, te - T)) & (t <= te)
    return th[s], x[s], v[s], om[s], pw[s]

rows = []
for nm, f in FILES:
    if not os.path.exists(f):
        print(f"缺文件 {f}"); continue
    th, x, v, om, pw = tail(f)
    dpw = np.diff(pw, prepend=pw[0])
    r_th = -W_TH * (th / TH_LIM) ** 2
    r_x  = -W_X  * (x  / X_LIM ) ** 2
    r_F  = -W_F  * (np.abs(pw) / MAXA) ** 2
    r_S  = -W_S  * (dpw / (2 * MAXA)) ** 2 * 4
    r    = R_SURV + r_th + r_x + r_F + r_S
    rows.append(dict(nm=nm, r=r.mean(),
                     th_t=r_th.mean(), x_t=r_x.mean(), F_t=r_F.mean(), S_t=r_S.mean(),
                     thstd=np.degrees(th.std()), pwstd=pw.std(),
                     omstd=om.std(), vstd=v.std(), xmean=x.mean()*1000))

D = pd.DataFrame(rows)

print("\n" + "="*94)
print("① 奖励逐项分解（真机数据，稳态段均值。R_SURVIVE=10 对所有人相同，不参与区分）")
print("="*94)
print(f"{'控制器':<11}{'总奖励':>9}{'θ项':>9}{'x项':>9}{'力项':>9}{'平滑项':>9}   |{'θstd°':>8}{'PWMstd':>8}")
print('-'*94)
for _, q in D.iterrows():
    print(f"{q.nm:<11}{q.r:9.3f}{q.th_t:9.4f}{q.x_t:9.4f}{q.F_t:9.4f}{q.S_t:9.4f}   |"
          f"{q.thstd:8.2f}{q.pwstd:8.1f}")

print("\n" + "="*94)
print("② 奖励排的名次 vs 真机实际好坏")
print("="*94)
D['rank_r']  = D.r.rank(ascending=False)          # 奖励越高越好
D['rank_th'] = D.thstd.rank()                      # theta std 越小越好
D['rank_mix']= (D.thstd.rank() + D.pwstd.rank())/2 # 复合基准
print(f"{'控制器':<11}{'奖励名次':>10}{'θstd名次':>10}{'复合名次':>10}{'奖励-θ 差':>11}")
print('-'*94)
for _, q in D.sort_values('rank_r').iterrows():
    print(f"{q.nm:<11}{q.rank_r:10.0f}{q.rank_th:10.0f}{q.rank_mix:10.1f}{q.rank_r-q.rank_th:+11.0f}")

rho_th,  p_th  = stats.spearmanr(D.r, -D.thstd)
rho_mix, p_mix = stats.spearmanr(D.r, -D.rank_mix)
print(f"\n奖励 vs θstd  秩相关 rho = {rho_th:+.3f}  (p={p_th:.5f})")
print(f"奖励 vs 复合   秩相关 rho = {rho_mix:+.3f}  (p={p_mix:.5f})")

print("\n" + "="*94)
print("③ 每一项【单独】能排出多好的名次 —— 谁在真正做区分")
print("="*94)
print(f"{'项':<12}{'与θstd秩相关':>14}{'p值':>10}{'跨控制器变化幅度':>18}")
print('-'*94)
for k, lab in [('th_t','θ项'), ('x_t','x项'), ('F_t','力项'), ('S_t','平滑项')]:
    rr, pp = stats.spearmanr(D[k], -D.thstd)
    span = D[k].max() - D[k].min()
    print(f"{lab:<12}{rr:+14.3f}{pp:10.5f}{span:18.4f}")

print("\n" + "="*94)
print("④ 假如加上速度项，名次会变好吗？（这是「该不该加」的直接证据）")
print("="*94)
OM_LIM, V_LIM = 3.0, 1.5     # 与仿真采样范围一致
base_rho = rho_th
print(f"{'方案':<34}{'秩相关':>10}{'相比现状':>12}")
print('-'*94)
print(f"{'现状（无速度项）':<34}{base_rho:+10.3f}{'—':>12}")
for W_OM in [1.0, 5.0, 20.0]:
    for W_V in [0.0, 5.0]:
        r2 = D.r - W_OM*(D.omstd/OM_LIM)**2 - W_V*(D.vstd/V_LIM)**2
        rr, _ = stats.spearmanr(r2, -D.thstd)
        tag = f"+ {W_OM:g}·(ω/3)²" + (f" + {W_V:g}·(v/1.5)²" if W_V else "")
        print(f"{tag:<34}{rr:+10.3f}{rr-base_rho:+12.3f}")

print("\n" + "="*94)
print("⑤ 反过来：现有五项里，去掉哪一项名次会崩？（等价于单项消融）")
print("="*94)
print(f"{'去掉':<14}{'秩相关':>10}{'相比现状':>12}")
print('-'*94)
for k, lab in [('th_t','θ项'), ('x_t','x项'), ('F_t','力项'), ('S_t','平滑项')]:
    r2 = D.r - D[k]          # 减去负值 = 去掉该惩罚
    rr, _ = stats.spearmanr(r2, -D.thstd)
    print(f"{lab:<14}{rr:+10.3f}{rr-base_rho:+12.3f}")
print()
