#!/usr/bin/env python3
"""校准仿真模型 v1（2026-10-03 定版）：以后训练、评估、写论文第 7 章都只认这一个版本。

用法：
    import s8_probe_sim as sim, sim_model_v1 as v1
    md = v1.md_v1()                       # 完整模型参数（可传关键字覆盖，比如 md_v1(_ctrl='lqr_factory')）
    r = sim.probe_sim(sim.FS, seed=1, BA=-0.02, md=md)

参数全部写死在下面（不读 npz），来源：
| 参数 | 值 | 来源 |
|---|---|---|
| Jp 摆杆转动惯量 | 0.00238 kg·m² | 自由摆动周期 1.060 s（swing.xlsx，swing_analyze.py） |
| dmp 转轴粘性阻尼 | 0.0006 N·m·s/rad | 自由摆动衰减（原厂 0.024 的约 1/40；合计等效 1/13–1/29） |
| tc 转轴库仑摩擦力矩 | 0.00095 N·m | 自由摆动：小摆幅衰减比例变大 |
| B 小车粘性阻尼 | 34.04 N·s/m | 推车实验（friction.xlsx，friction_analyze.py） |
| fcp / fcn 小车库仑摩擦 | 1.445 / 1.532 N | 推车实验（正向/负向） |
| fs 小车静摩擦 | 2.0 N | 推车实验（拟合顶在 2.0；慢斜坡起步 60–120 PWM） |
| kf 摩擦在平衡工况下的系数 | 0.8295 | 混合拟合（S8 数据） |
| ka 执行器增益 | 1.647（推力 = ka·PWM/42） | 混合拟合；推车实验独立测得 1.640 |
| fa 执行器一阶滞后 | 4.570 Hz | 混合拟合（推车实验 4.11 Hz） |
| ta 执行器纯延迟 | 5.85 ms | 混合拟合 |
| tal 角度测量延迟 | 19.91 ms | 混合拟合（物理来源在下位机/传感器一侧，Simulink 里没有） |
| 角度读数噪声 | 回放 angle_noise_hw.npz | S8 静置段 >15 Hz 残差，std 3.8–4.5 mrad |
| 角度量化 | 1.538 mrad | 真机 12 位 |
| dist 过程噪声 | 4.076 | 混合拟合（对上 S8 静置摆角 std 1.30°） |
| 观测链 | 1 Hz / 0.5 Hz 二阶微分器，ω 为真值 2 倍 | 与部署链路一致（OMEGA_SCALE） |
混合拟合：sim_hybrid_fit2.py，目标只用 S8 两次探测的 G、|S| 与 S8 静置 std。
检验（都不参与拟合，详见 logs/model_v1/model_card.md）：原厂 LQR 块（10/02）八点全在 ±20%；
9/07–9/08 七个控制器 55 点中位 0.99、82% 在 ±20%；9/21 五个网络 C、E 起振、S8 稳。
"""
import numpy as np
import s8_probe_sim as sim

V1 = dict(Jp=0.00238, dmp=0.0006, tc=0.00095,
          B=34.04206067000318, fcp=1.4445244283469572, fcn=1.532274630479637, fs=2.0, kf=0.8295060174330753,
          ka=1.6469698769364034, fa=4.569647799893637, ta=0.005848074804216543, tal=0.019911386105418448)
DIST_V1 = 4.076189447702792
Q_TH_V1 = 1.538e-3
_NZ = np.load('angle_noise_hw.npz')
NOISE_V1 = np.r_[_NZ['r2'], _NZ['r1']]


def md_v1(**overrides):
    """完整的 v1 参数字典（交给 s8_probe_sim.probe_sim 的 md）"""
    md = dict(sim.MD); md.update(V1)
    md.update(_noise_seq=NOISE_V1, _q_th=Q_TH_V1, _dist=DIST_V1)
    md.update(overrides)
    return md


if __name__ == '__main__':
    # 自检：与混合拟合结果一致
    import sim_hybrid_fit2 as h2
    P = np.load('sim_hybrid_params2.npz'); old = h2.md_of([float(P[k]) for k in h2.NAMES]); new = md_v1()
    diff = {k: (old[k], new[k]) for k in set(old) | set(new) if k != '_noise_seq' and not np.allclose(old.get(k, np.nan), new.get(k, np.nan))}
    print('与 sim_hybrid_params2 + 实测参数逐项一致' if not diff else f'不一致：{diff}')
    r1 = sim.probe_sim([2.5], seed=1, TW=25, md=old)['S'][0]; r2 = sim.probe_sim([2.5], seed=1, TW=25, md=new)['S'][0]
    print(f'同一种子 2.5 Hz |S|：{r1:.4f} vs {r2:.4f}')
