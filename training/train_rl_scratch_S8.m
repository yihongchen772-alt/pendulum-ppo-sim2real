%**************************************************************************
% RL 训练 S3 —— 从零训练线（Scratch line）第三版
%
% 【命名】本项目分两条线：
%   热启动线：v20 -> v24 -> v25b（已定稿，从 LQR 克隆网络热启动）
%   从零线  ：S1(原v22) -> S2 -> S3(本版)（随机初始化，不用 LQR 权重）
%
% 【S1 的真机结果与失败原因】
%   theta 1.96x / PWM 2.29x / 偏 197mm —— 三项全垫底
%   查明原因：自由探索跑到了结构的错误角落
%     Kalpha  -17.40 (克隆 -8.80)   角度增益翻倍
%     Komega  -12.15 (克隆 -22.06)  角速度砍半
%     Ka/Kx     7.44 (克隆   4.22)  导致偏 197mm
%   而真机上角度通道在4Hz处100%通过、角速度只有1.6% —— S1 恰好选了最差的配比
%
% 【S2 的真机结果：失败】2026-09-03 实测
%   主频 2.80Hz(应3.95) / theta std 0.0433(2.20x) / PWM std 288(2.20x)
%   偏 -233mm（史上最差，S1 为 -197mm）/ 41 秒撞限位
%   根因：z0 = mS*0 使四条约束退化为【单点】约束。优化器在原点竖一根尖针
%   满足全部比例，其余地方随意：alpha=0 时 Ka=20.12，alpha=0.01 即掉到 5.45。
%   而真机常驻 alpha≈0.02，全在约束管不到的地方。日志显示的 Kw/Ka=2.51、
%   Ka/Kx=3.79 全部"达标"，但那只是原点值，工作点实际是 7.43 / 2.20。
%   另：四条约束中无一条管"推力下限"（Gamma 是上限铰链，比例只管配比不管
%   幅度），故"处处偏软"这一失败模式完全不在覆盖范围内——S2 在 x=-0.23 处
%   推力仅为克隆的 22%，拉不回中心。
%
% 【S3 的改动】
%   ① 五条约束全部改在【每次重采样的 16 个随机锚点】上计算
%      （x∈±0.15, alpha∈±0.03, v=omega=0 以避开 tanh 饱和区）
%      注意：偏置约束 ④ 必须仍只在原点算，否则会强制 mu(x=±0.15)=0，
%      与"远场要有回复力"直接冲突。
%   ② 新增第五条：远场推力下限（相对克隆 ≥70%，铰链形式）
%   ③ calcMetrics 同步改到锚点上算——否则日志会重蹈 S2 覆辙、报喜不报忧
%   ④ GAMMA_REF 在锚点分布上重新标定：10.381 -> 10.290
%
% 【S2 加入的四条【经真机对照实验验证】的结构约束】：
%   Komega/Kalpha ≈ 2.51  -> 极限环频率守在 3.95Hz  (v24改到5.33->2.21Hz; v25b锁2.51->3.96Hz)
%   Kalpha/Kx     ≈ 4.22  -> 居中                  (S1为7.44->偏197mm; v24/v25b锁4.2->偏40mm)
%   Gamma(4Hz)    ≤ 0.80x -> 抖动                  (1.00/1.18/1.78 -> 抖动1.00/1.44/1.96)
%   原点偏置      ≈ 0      -> 防静态偏置            (v19 +0.207 等效常驻165PWM)
%
% 【说明】这些约束的【机制】来自真机测量，其【目标数值】取自项目 LQR 设计。
%   因此 S2 不是"零先验"，而是"不用 LQR 权重初始化，但结构受真机验证的约束引导"。
%
% 【关键实现细节】约束权重随难度课程一起渐进施加（乘 cfg.diff）。
%   随机初始化时增益接近0，比值类约束会产生巨大惩罚、直接压死探索——
%   这正是 v21 全难度一把梭 2320 局卡在 450 步的教训。
% v21 的教训：六个维度（滤波滞后/量化/域随机化/持续扰动/冲击/重平滑惩罚）
% 从第一步就全开，随机策略被平滑惩罚(单步-24，比生存奖励+10还大)压死在
% 探索阶段，2300局仍卡在450步。v22 让难度本身也走课程：先在接近 v3 的
% 宽松环境里学会基本平衡，再逐步把六项难度加满。
% 与 v20 的唯一区别：不做热启动、不加锚定惩罚，改用课程学习从随机初始化练起。
% 用来回答：不靠模仿 LQR，纯 RL 能不能自己学出一个可用（甚至更好）的控制器。
%
% 跟 v7~v19 的根本区别（那些版本全部真机失败）：
%   1. 观测链路照搬真机：角度/位置量化 -> 二阶微分滤波器(1Hz车速 / 0.5Hz角速度)
%      -> 角速度再 x2（真机实测的放大，是承重的，不能"修"）
%      不再用"自己差分 + 一阶EMA"那套（那是 v13~v19 用的，跟真机对不上）
%   2. 摆杆转动惯量 Jp 用扫频实测辨识值 0.0025（原代码 0.01 大了4倍）
%   3. MAX_ACTION = 1000（对齐 Saturation2 真实限幅，原来用 800）
%   4. 域随机化：Jp / 角速度倍数 / PWM->力系数 / 滤波器带宽 每局随机，
%      策略不许精确拟合到某一个可能错的值上
%   5. 【本线不做热启动】随机初始化，不加锚定惩罚 —— 这是与 v20 热启动线的
%      唯一区别（见上文第 52 行）。cloneNet 只作参照值与基准，不参与训练。
%      对照：v19 从 LQR 热启动，PPO 把增益放大 2.4 倍 + 长出 0.207 静态偏置，
%      真机最抖 —— 热启动并不保证继承 LQR 的好性质。
%   7. 断点按"可部署性"选，不按仿真存活步数
%      （v19 仿真5000满分却是真机最差的，说明仿真步数不是有效指标）
%
% 目标（用户指定，三个都要）：
%   A 更省力更平滑  -> r_force / r_smooth 加重
%   B 抗扰动更强    -> 训练中随机施加冲击力
%   C 小车守住中心  -> r_x 加重 + 随机初始偏置
%**************************************************************************
clc; clear; close all;

%% ---------------- 标定后的真机参数（全部有实测依据）----------------
P.DT         = 0.004;
P.MAX_ACTION = 1000;          % Saturation2 真实限幅
P.Q_TH       = (360/2048)*pi/180;   % 角度量化 0.00307 rad
P.Q_X        = 0.658/5268;          % 位置量化 0.000125 m
P.THETA_LIMIT= 0.2;
P.X_LIMIT    = 0.2;

% 标称值（域随机化围绕它们波动）
P.Jp_nom   = 0.0025;   % 扫频辨识值（=均匀杆理论值 mL^2/12）
P.OM_nom   = 2.0;      % 角速度信号放大倍数（实测 1.995）
P.P2F_nom  = 42;       % PWM->力：F = pwm/P2F（实测标定值）
P.FV_nom   = 1.0;      % 车速滤波器带宽倍数（标称 1 Hz）
P.FO_nom   = 0.5;      % 角速度滤波器带宽（Hz）

% 域随机化范围
P.Jp_rng  = [0.0020 0.0120];
P.OM_rng  = [1.60 2.40];
P.P2F_rng = [36 50];
P.FO_rng  = [0.40 0.65];
P.FV_rng  = [0.80 1.30];

%% ---------------- 超参数 ----------------
cfg.ACTOR_LR   = 1e-4;      % 从零探索，需要更大步长
cfg.CRITIC_LR  = 5e-4;
cfg.GAMMA      = 0.99;
cfg.LAMBDA     = 0.95;
cfg.CLIP_EPS   = 0.2;
ENTROPY_BETA   = 5e-3;
K_EPOCHS       = 8;
BATCH_SIZE     = 256;
MAX_EPISODES   = 30000;
cfg.MAX_STEPS  = 3000;      % 12 秒/局
cfg.cur_theta  = 0.35;      % 课程学习：起始宽松边界
cfg.diff       = 0.0;
% ---- 四条经真机验证的结构约束（目标值取自 LQR 设计）----
cfg.WRATIO_REF = 2.51;   % Komega/Kalpha -> 极限环频率
cfg.RATIO_REF  = 4.22;   % Kalpha/Kx     -> 居中
cfg.GAMMA_REF  = 10.290; % （保留，仅用于日志对照；损失已不再使用）
% ============ 【S5】Gamma 改用【真机验证过的时域口径】 ============
% S4 的教训：训练用的"方向导数代理"只对线性网络准（克隆偏差 0.1%），
% 对非线性网络误差 27%~61% 且方向不定：
%   v25b 代理0.76x/真值0.60x | S2 1.32/0.93 | S3 0.60/1.15 | S4 0.56/1.42
% 数学原因：时域 Gamma 是【描述函数】（对有限幅正弦的基波响应），对非线性
% 环节依赖幅度；代理是某一点的方向导数。线性时二者相等，非线性时无此保证。
% 由于输入轨迹不依赖网络，可预计算，则时域 Gamma 全程可微，可直接进损失。
cfg.GTD_F   = 3.95;    % 真机实测极限环频率 Hz
cfg.GTD_T   = 4.0;     % 窗长秒（3.95Hz 下约 16 个周期，足够单频提取）
% 【S8】Gamma 改在【一组振幅】上评估。S7 的教训：cfg.GTD_A 写死 0.02，
% 网络在该振幅达标（0.80x）而实际工作在 0.093 rad 处为 1.08x，且增益随
% 振幅【上升】(0.71->1.18) 构成正反馈 —— 真机 theta 抖动 5.33 度，5 倍于 LQR。
% 描述函数本就依赖振幅，只在一个振幅上约束等于没约束。
cfg.GTD_AMPS = [0.010 0.020 0.040 0.060 0.090];   % 覆盖真机可能的工作振幅
cfg.W_GRISE  = 20.0;   % 禁止 Gamma 随振幅上升的惩罚权重
cfg.GAMMA_TGT  = 0.80;   % Gamma 目标（相对基准）
cfg.GAMMA_DIR  = [0;0;1;0.0785];
cfg.GAMMA_DEL  = 0.02;
cfg.W_WRATIO   = 12.0;
cfg.W_RATIO    = 12.0;
cfg.W_GAMMA    = 10.0;
cfg.W_BIAS     = 20.0;       % 【v22】难度系数 0->1，控制扰动/随机化/平滑惩罚

% ============ 【S3】相对 S2 的两处修正 ============
% S2 真机失败原因：z0 = mS*0 使四条约束退化为【单点】约束，优化器只在原点
% 竖起一根尖针满足全部比例，离开 0.01 rad 增益即掉到 27%。真机常驻 alpha≈0.02、
% 偏移几十毫米，全在约束管不到的地方 -> 实测偏 -233mm、41 秒撞限位。
%
% 修正①：约束改在【每次重采样的随机锚点】上计算，尖峰无处可藏。
%         固定锚点仍可能被针对性满足，故每次更新重采样。
cfg.NANCH      = 16;         % 锚点个数
cfg.ANCH_X     = 0.30;       % 锚点 x 范围（±0.15 m）
cfg.ANCH_A     = 0.06;       % 锚点 alpha 范围（±0.03 rad）
%
% 修正②：新增第五条约束——远场推力下限。
%         原四条中，Gamma 是上限铰链（太大才罚），比例约束只管配比不管幅度，
%         没有任何一条防止"处处偏软"。S2 在 x=-0.23 处推力仅为克隆的 22%，
%         直接导致拉不回中心。参照值取自克隆网络实测。
cfg.AUTH_PTS   = [-0.23 -0.10  0.10  0.23  0.00  0.00  0.00  0.00;
                   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00;
                   0.00  0.00  0.00  0.00  0.10 -0.10  0.04 -0.04;
                   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00];
% 【S4】改为【带符号】参照值。S3 的教训：用 |mu| 会让"用力推错方向"的网络
% 拿到高分——S3 在 x=-0.23 处输出 +0.725（克隆是 -0.467，方向相反），
% 旧口径读数 |mu|/REF = 1.55 判为"达标"，实际真机上把小车往外推。
cfg.AUTH_REF   = [-0.4673 -0.2004 +0.2109 +0.4725 -0.8502 +0.8707 -0.3389 +0.3431];
% 阈值 0.50 由四个【已知真机结论】的网络标定（逐点带符号比值的最小值）：
%   克隆 1.00 成功 / v25b 0.59 成功 / S2 0.27 失败 / S3 -1.55 失败
%   取 0.50 使前两者通过、后两者拒绝，v25b 余量 0.09
% 【S6】推力约束由【单边下限】改为【双边区间】。
% S5 教训：只有下限、没有上限，网络学成 bang-bang 照样"达标"
%   （S5 八点比值 2.14/3.86/4.50/2.12/1.15/1.13/1.48/1.90，|x|>=100mm 即打满）
% 上限依据（2026-09-04 真机抗扰对照）：
%   克隆x0.85（比值恒 1.00）：推动后稳定 1.39s（中位 0.90s），小车行程 49mm
%   v25b（比值 0.59~0.94）：  稳定 2.30s，小车行程 106mm，PWM 峰 918
%   => "远场增益越大抗扰越强"在真机上【不成立】，不该鼓励大增益。
% 【S8】区间整体下移。依据（2026-09-07 三次步进正弦实测灵敏度）：
%   克隆x1.00  Ms=5.59   v25b Ms=5.27   克隆x0.75 Ms=4.13
%   => 改策略【形状】动不了 Ms；降整体【增益】能降 26%。
%   而旧区间 [0.55,1.40] 的下限反把网络往高增益推（S7 顶到 0.98~1.36），
%   正是其 Ms 差、抖动大的原因之一。现主动往低增益走。
cfg.AUTH_LO    = 0.45;   % v25b 实测 0.59（真机成功），留 0.04 余量；S2 0.27/S3 -1.55 失败
cfg.AUTH_HI    = 0.85;   % 【损失】训练目标：主动往低增益走
% 【门槛】用另一个更松的上限。区分两件事：
%   AUTH_HI     = 0.85  是【设计偏好】（低增益 -> 低 Ms），放在损失里当目标
%   AUTH_HI_GATE= 1.40  是【失效判据】，用真机失败标定：
%                       S5 4.50(bang-bang) / S2 3.38 / S4 1.91 均失败
%                       克隆 1.00 / v25b 0.94 / S6 1.36 均可用
% 门槛若用 0.85，会把真机验证成功的 v25b(0.94) 一并拒掉 —— 那是把偏好
% 当成了判据。断点该在"不会失效"的集合里选，而不是只收"完全符合偏好"的。
cfg.AUTH_HI_GATE = 1.40;   % S5 最高 4.50、S4 1.91 均失败；克隆 1.00、v25b 0.94 通过
cfg.AUTH_TGT   = cfg.AUTH_LO;   % 兼容旧名
cfg.W_AUTH     = 15.0;
% 【S4】通道符号：由 LQR 解析式与克隆网络双重确认
%   dmu/dx > 0    (du/dx = -K1 = +38.73)
%   dmu/dalpha < 0 (du/dalpha = -K2 = -164.21)
%   dmu/domega < 0 (du/domega = -K4 = -41.06)
cfg.SGN        = [+1, 0, -1, -1];   % 各通道期望符号（0=不约束）
cfg.W_SGN      = 25.0;
% =================================================
cfg.cur_x      = 0.30;
ROLLOUT_EP     = 8;

cfg.LOG_STD_OFFSET = 1.5;   % 【v22】初始 sigma 0.74 -> 0.22，避免随机bang-bang
cfg.SIGMA_MIN      = 0.02;
cfg.ANCHOR_W       = 0.0;   % 【v21】不锚定

% 奖励权重（A/B/C）
cfg.W_THETA  = 5.0;
cfg.W_X      = 8.0;    % C：加重居中
cfg.W_FORCE  = 1.5;    % A：加重省力
cfg.W_SMOOTH = 6.0;    % A：大幅加重平滑（这是压住4Hz抖动的主力）
cfg.R_SURVIVE= 10.0;
cfg.R_FAIL   = -50.0;
cfg.DIST_STD = 2.0;    % 持续过程扰动(rad/s^2)，让"稳住"本身有难度
cfg.KICK_MAG = 15.0;   % B：冲击强度

ACTOR_GN = 0.5; CRITIC_GN = 5.0;

%% ---------------- 网络：随机初始化（不热启动）----------------
% 【注意】cloneNet 只用于：算 Gamma/推力约束的参照值、打印基准成绩。
% 它【不参与训练】—— actorNet 是 dlnetwork(actorLayers) 纯随机初始化，
% 全脚本没有任何地方把克隆权重赋给 actorNet，ppoGrad 里 cloneNet 也未被使用。
S = load('actor_net_lqr_clone.mat');
cloneNet  = S.actorNet;      % 仅作最终对比基准，不参与训练
actorLayers = [
    featureInputLayer(4,'Normalization','none','Name','in')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(2,'Name','out')
];
actorNet = dlnetwork(actorLayers);   % 【v21】随机初始化

criticLayers = [
    featureInputLayer(4,'Normalization','none','Name','in')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(1,'Name','v')
];
criticNet = dlnetwork(criticLayers);

% ---- 【S5】预计算时域 Gamma 的输入轨迹（不依赖网络，一次性）----
[gb,ga_] = diffFilt(P.FO_nom, P.DT);
Ng = round(cfg.GTD_T/P.DT); tg = (0:Ng-1)'*P.DT;
h0 = round(Ng*0.5);                      % 丢掉前半段暂态
nA = numel(cfg.GTD_AMPS); Lw = Ng-h0+1;
Oall = zeros(4, nA*Lw);
for ia = 1:nA
    thg = cfg.GTD_AMPS(ia)*sin(2*pi*cfg.GTD_F*tg);
    uu=zeros(1,3); yy=zeros(1,3); Og=zeros(4,Ng);
    for n=1:Ng
        alg=-thg(n); uu=[alg uu(1:2)];
        yy=[(gb(1)*uu(1)+gb(2)*uu(2)+gb(3)*uu(3)-ga_(2)*yy(1)-ga_(3)*yy(2)), yy(1:2)];
        Og(:,n)=[0;0;alg;P.OM_nom*yy(1)/10];
    end
    Oall(:,(ia-1)*Lw+1:ia*Lw) = Og(:,h0:end);
end
cfg.GTD_O   = dlarray(single(Oall),'CB');            % 5 段拼接，一次前向
cfg.GTD_C   = single(cos(2*pi*cfg.GTD_F*tg(h0:end))');
cfg.GTD_S   = single(sin(2*pi*cfg.GTD_F*tg(h0:end))');
cfg.GTD_LW  = Lw; cfg.GTD_NA = nA;
cfg.GTD_REF = calcGtd(cloneNet, cfg);                % 逐振幅的克隆基准（向量）
fprintf('【S8】Gamma 基准（克隆，逐振幅）:'); fprintf(' %.2f',cfg.GTD_REF); fprintf('\n');

fprintf('【S8】随机初始化 + 难度课程 + 五条约束（Gamma 改用真机时域口径）\n');

%% ---------------- 先测一下起点 ----------------
fprintf('\n--- 基准（克隆网络，仅供对比）---\n');
b = evaluatePolicy(cloneNet, P, cfg, 20, true);
printEval(b, '克隆网络');
base_score = b.score;

%% ---------------- PPO 微调 ----------------
avgGA=[]; avgSGA=[]; avgGC=[]; avgSGC=[]; gstep=0;
hist_steps=[]; total_ep=0; iter=0;
best_score = -inf; best_iter = 0;
lossA=dlarray(0); lossC=dlarray(0);

NET_OUT  = 'actor_net_S8.mat';
NET_BEST = 'actor_net_S8_best.mat';


fprintf('\n开始从零 PPO 训练（课程学习）...\n');
while total_ep < MAX_EPISODES
    iter = iter+1;
    aS=[]; aA=[]; aL=[]; aT=[]; aAdv=[];
    for k=1:ROLLOUT_EP
        if total_ep>=MAX_EPISODES, break; end
        total_ep = total_ep+1;
        [st,ac,lp,adv,td,T] = runEpisode(actorNet, criticNet, P, cfg);
        aS=[aS st]; aA=[aA ac]; aL=[aL lp]; aT=[aT td]; aAdv=[aAdv adv];
        hist_steps(end+1)=T; %#ok<AGROW>
    end
    N=numel(aA);
    if N>5, aAdv=(aAdv-mean(aAdv))/(std(aAdv)+1e-8); end
    ep_n = K_EPOCHS; if N<64, ep_n=2; end
    for e=1:ep_n
        pidx=randperm(N);
        for s0=1:BATCH_SIZE:N
            e0=min(s0+BATCH_SIZE-1,N); bi=pidx(s0:e0);
            mS=dlarray(aS(:,bi),'CB'); mA=dlarray(aA(bi),'CB');
            mL=dlarray(aL(bi),'CB');   mT=dlarray(aT(bi),'CB');
            mAd=dlarray(aAdv(bi),'CB');
            gstep=gstep+1;
            [lossA,lossC,gA,gC]=dlfeval(@ppoGrad, actorNet, criticNet, cloneNet, ...
                mS,mA,mL,mT,mAd, ENTROPY_BETA, cfg);
            gA=clipGN(gA,ACTOR_GN); gC=clipGN(gC,CRITIC_GN);
            [actorNet,avgGA,avgSGA]=adamupdate(actorNet,gA,avgGA,avgSGA,gstep,cfg.ACTOR_LR);
            [criticNet,avgGC,avgSGC]=adamupdate(criticNet,gC,avgGC,avgSGC,gstep,cfg.CRITIC_LR);
        end
    end

    % 课程学习门控：最近30局平均步数达标才收紧
    rm = mean(hist_steps(max(1,end-29):end));
    if rm >= cfg.MAX_STEPS*0.35
        cfg.cur_theta = max(P.THETA_LIMIT, cfg.cur_theta - (0.35-P.THETA_LIMIT)/20);
        cfg.cur_x     = max(P.X_LIMIT,     cfg.cur_x     - (0.30-P.X_LIMIT)/20);
        cfg.diff      = min(1.0, cfg.diff + 0.02);      % 【v22】达标才加难度
    elseif rm < cfg.MAX_STEPS*0.15
        cfg.diff      = max(0.0, cfg.diff - 0.01);      % 掉太狠就回退难度
    end
    if mod(iter,10)==0
        ev = evaluatePolicy(actorNet, P, cfg, 20, true);
        flag='';
        % 【S8】门槛与损失保持一致：已删除 Kw/Ka 与 Ka/Kx 两条（纯 LQR 模仿，
        % 且 Kw/Ka 决定频率的说法已被 S7 证伪）。保留的均与 LQR 无关：
        ok_shape = (cfg.diff > 0.9) ...
                && ev.gratio <= 1.00 ...          % 全振幅最大 Gamma。真机标定：
                ...                               %   v25b 0.87(成功)/S6 0.92(跑通)/S7 1.18(失败)
                && ev.grise  <= 0.30 ...          % Gamma 不得随振幅上升。真机标定：
                ...                               %   v25b +0.28/S6 +0.09/S7 +0.38(失败)
                && abs(ev.mu0) < 0.03 ...         % 原点无静态偏置
                && ev.auth   >= cfg.AUTH_LO ...   % 远场推力下限（防 S2/S3 拉不回中心）
                && ev.authmax <= cfg.AUTH_HI_GATE ...  % 上限：防 bang-bang（S5 4.50/S2 3.38 失败）
                && ev.sgnok;                      % 三通道符号正确（防 S3 推反方向）
        if ok_shape && ev.score > best_score
            best_score = ev.score; best_iter=iter;
            save(NET_BEST,'actorNet'); flag=' *** 新最佳，已保存';
        end
        fprintf(['Ep %5d | 训练存活 %6.1f | 评估存活 %6.1f | mu0 %+.3f | ' ...
                 'Gamma %.2fx(升%+.2f) | Kw/Ka %4.2f | Ka/Kx %4.2f | 推力 %.2f~%.2f | 符号 %d | 难度 %.2f | 得分 %6.2f%s\n'], ...
            total_ep, rm, ev.steps, ev.mu0, ev.gratio, ev.grise, ev.wratio, ev.ratio, ev.auth, ev.authmax, ev.sgnok, cfg.diff, ev.score, flag);
    end
end

save(NET_OUT,'actorNet');
fprintf('\n训练结束。最佳出现在 iter %d，得分 %.2f（起点 %.2f）\n', best_iter, best_score, base_score);
fprintf('最佳网络 -> %s\n', NET_BEST);

fprintf('\n=== 最终对比 ===\n');
Sc=load(NET_BEST);
printEval(evaluatePolicy(cloneNet,   P,cfg,40,true), '克隆网络(起点)');
printEval(evaluatePolicy(Sc.actorNet,P,cfg,40,true), 'S2 最佳');
fprintf('\n--- 抗扰动测试（施加冲击）---\n');
printEval(evaluatePolicy(cloneNet,   P,cfg,40,true), '克隆网络(起点)');
printEval(evaluatePolicy(Sc.actorNet,P,cfg,40,true), 'S2 最佳');


%% ======================= 子函数 =======================
function pr = sampleParams(P,d)
    if nargin<2, d=1.0; end
    % 【v22】难度系数 d 控制随机化宽度：d=0 时全部取标称值，d=1 时全宽
    r=@(a,n) n + d*((a(1)+(a(2)-a(1))*rand()) - n);
    pr.Jp=r(P.Jp_rng,P.Jp_nom); pr.OM=r(P.OM_rng,P.OM_nom); pr.P2F=r(P.P2F_rng,P.P2F_nom);
    pr.FO=r(P.FO_rng,P.FO_nom); pr.FV=r(P.FV_rng,P.FV_nom);
    [pr.bo,pr.ao]=diffFilt(pr.FO,P.DT);
    [pr.bv,pr.av]=diffFilt(pr.FV,P.DT);
end

function [b,a]=diffFilt(f0,DT)
    % 二阶微分滤波器 wn^2*s/(s^2+2*z*wn*s+wn^2)，Tustin 离散化（手写，不依赖工具箱）
    z=0.9; wn=2*pi*f0; K=2/DT;
    % 分子 wn^2*s -> wn^2*K*(1-z^-1)/(1+z^-1)
    % 分母 s^2+2z wn s+wn^2
    b0=wn^2*K; b1=0; b2=-wn^2*K;
    a0=K^2+2*z*wn*K+wn^2;
    a1=2*(wn^2-K^2);
    a2=K^2-2*z*wn*K+wn^2;
    b=[b0 b1 b2]/a0; a=[1 a1/a0 a2/a0];
end

function [obs,stt]=makeObs(theta,x,stt,pr,P)
    th_q=round(theta/P.Q_TH)*P.Q_TH;
    x_q =round(x    /P.Q_X )*P.Q_X;
    al = -th_q;                       % LQR 约定：alpha = -theta
    stt.ua=[al  stt.ua(1:2)];
    stt.ya=[(pr.bo(1)*stt.ua(1)+pr.bo(2)*stt.ua(2)+pr.bo(3)*stt.ua(3) ...
            -pr.ao(2)*stt.ya(1)-pr.ao(3)*stt.ya(2)), stt.ya(1:2)];
    stt.ux=[x_q stt.ux(1:2)];
    stt.yx=[(pr.bv(1)*stt.ux(1)+pr.bv(2)*stt.ux(2)+pr.bv(3)*stt.ux(3) ...
            -pr.av(2)*stt.yx(1)-pr.av(3)*stt.yx(2)), stt.yx(1:2)];
    omega = pr.OM*stt.ya(1);          % 真机的 2 倍放大
    obs=[x_q; stt.yx(1)/5.0; al; omega/10.0];
end

function stt=initFilt()
    stt.ua=zeros(1,3); stt.ya=zeros(1,3);
    stt.ux=zeros(1,3); stt.yx=zeros(1,3);
end

function ds=dyn(st,pwm,pr)
    Mc=0.4719; Mp=0.1679; l=0.2125; g=9.81; dmp=0.024;
    xd=st(2); th=st(3); thd=st(4);
    F=pwm/pr.P2F; ml=Mp*l; ml2=Mp*l^2;
    M=[Mc+Mp, ml*cos(th); ml*cos(th), pr.Jp+ml2];
    Fv=[F-30*xd+ml*thd^2*sin(th); -dmp*thd+Mp*g*l*sin(th)];
    a=M\Fv; ds=[xd;a(1);thd;a(2)];
end

function [states,acts,logps,adv,td,T]=runEpisode(actorNet,criticNet,P,cfg)
    pr=sampleParams(P,cfg.diff);
    theta=(rand()-0.5)*0.06; thd=0;
    x=(rand()-0.5)*0.10; xd=0;                 % C：随机初始偏置
    stt=initFilt();
    MS=cfg.MAX_STEPS;
    states=zeros(4,MS); acts=zeros(1,MS); logps=zeros(1,MS);
    rew=zeros(1,MS); dones=zeros(1,MS); vals=zeros(1,MS);
    prev_pwm=0; steps=0; done=false;
    kick_at = randi([200, MS]);                % B：随机时刻施加冲击
    while ~done && steps<MS
        steps=steps+1;
        [obs,stt]=makeObs(theta,x,stt,pr,P);
        states(:,steps)=obs;
        dl=dlarray(obs,'CB');
        oa=extractdata(predict(actorNet,dl));
        vals(steps)=extractdata(predict(criticNet,dl));
        mu_raw=oa(1);
        lstd=min(max(oa(2)-cfg.LOG_STD_OFFSET,-20),2);
        sig=max(exp(lstd),cfg.SIGMA_MIN);
        z=mu_raw+sig*randn(); a_t=tanh(z);
        logps(steps)=-0.5*((z-mu_raw)^2)/sig^2 - log(sig) - 0.5*log(2*pi) - log(1-a_t^2+1e-6);
        acts(steps)=z;
        pwm=a_t*P.MAX_ACTION;

        d4=dyn([x;xd;theta;thd],pwm,pr);
        d4(4)=d4(4)+cfg.diff*cfg.DIST_STD*randn();              % 持续扰动(随难度)
        if steps==kick_at, d4(4)=d4(4)+cfg.diff*(rand()-0.5)*2*cfg.KICK_MAG; end   % B：冲击(随难度)
        xn=x+P.DT*d4(1); xdn=xd+P.DT*d4(2);
        thn=theta+P.DT*d4(3); thdn=thd+P.DT*d4(4);

        if abs(thn)>cfg.cur_theta || abs(xn)>cfg.cur_x
            r=cfg.R_FAIL; done=true;
        else
            r = cfg.R_SURVIVE ...
              - cfg.W_THETA *(thn/P.THETA_LIMIT)^2 ...
              - cfg.W_X     *(xn /P.X_LIMIT)^2 ...
              - cfg.W_FORCE *(abs(pwm)/P.MAX_ACTION)^2 ...
              - (0.3+0.7*cfg.diff)*cfg.W_SMOOTH*((pwm-prev_pwm)/(2*P.MAX_ACTION))^2*4;  % 平滑惩罚随难度
            done=false;
        end
        rew(steps)=r; dones(steps)=done; prev_pwm=pwm;
        x=xn; xd=xdn; theta=thn; thd=thdn;
    end
    T=steps;
    states=states(:,1:T); acts=acts(1:T); logps=logps(1:T);
    rew=rew(1:T); dones=dones(1:T); vals=vals(1:T);
    [obsf,~]=makeObs(theta,x,stt,pr,P);
    vlast=extractdata(predict(criticNet,dlarray(obsf,'CB')));
    ve=[vals vlast];
    adv=zeros(1,T); g=0;
    for t=T:-1:1
        dlt=rew(t)+cfg.GAMMA*ve(t+1)*(1-dones(t))-ve(t);
        g=dlt+cfg.GAMMA*cfg.LAMBDA*(1-dones(t))*g;
        adv(t)=g;
    end
    td=adv+vals;
end

function ev=evaluatePolicy(net,P,cfg,NEP,withKick)
    if nargin<5, withKick=true; end
    steps=zeros(1,NEP); mus=[]; pwms=[]; xstd=zeros(1,NEP); rec=zeros(1,NEP);
    for e=1:NEP
        rng(1000+e);  %#ok<RAND>
        pr=sampleParams(P,0);
        pr.Jp=P.Jp_nom; pr.OM=P.OM_nom; pr.P2F=P.P2F_nom;   % 评估用标称值
        [pr.bo,pr.ao]=diffFilt(P.FO_nom,P.DT); [pr.bv,pr.av]=diffFilt(P.FV_nom,P.DT);
        theta=(rand()-0.5)*0.06; thd=0; x=(rand()-0.5)*0.08; xd=0;
        stt=initFilt(); MS=cfg.MAX_STEPS; alive=MS;
        mu_l=zeros(1,MS); pw_l=zeros(1,MS); x_l=zeros(1,MS);
        kick_at=round(MS*0.4); recov=MS;
        for t=1:MS
            [obs,stt]=makeObs(theta,x,stt,pr,P);
            oa=extractdata(predict(net,dlarray(obs,'CB')));
            mu=tanh(double(oa(1))); pwm=mu*P.MAX_ACTION;
            mu_l(t)=mu; pw_l(t)=pwm; x_l(t)=x;
            d4=dyn([x;xd;theta;thd],pwm,pr);
            d4(4)=d4(4)+cfg.DIST_STD*randn();
            if withKick && t==kick_at, d4(4)=d4(4)+cfg.KICK_MAG; end
            x=x+P.DT*d4(1); xd=xd+P.DT*d4(2);
            theta=theta+P.DT*d4(3); thd=thd+P.DT*d4(4);
            if withKick && t>kick_at && recov==MS && abs(theta)<0.02, recov=t-kick_at; end
            if abs(theta)>P.THETA_LIMIT||abs(x)>P.X_LIMIT, alive=t; break; end
        end
        steps(e)=alive; m=min(500,alive):alive;
        mus=[mus mu_l(m)]; pwms=[pwms pw_l(m)]; xstd(e)=std(x_l(m)); rec(e)=recov;
    end
    o0=extractdata(predict(net,dlarray([0;0;0;0],'CB')));
    ev.steps=mean(steps); ev.mu_std=std(mus); ev.mu0=tanh(double(o0(1)));
    ev.pwm_std=std(pwms); ev.x_std=mean(xstd); ev.recov=mean(rec)*P.DT;
    ev.sat=mean(abs(mus)>0.9)*100;
    % 可部署性得分：存活为主，抖动/偏置/居中为辅（越高越好）
    mm = calcMetrics(net,cfg);
    ev.ratio=mm.ratio; ev.wratio=mm.wratio; ev.gratio=mm.gratio; ev.Kx=mm.Kx; ev.auth=mm.auth; ev.authmax=mm.authmax; ev.sgnok=mm.sgnok; ev.wdev=mm.wdev; ev.grise=mm.grise;
    ev.score = (ev.steps/cfg.MAX_STEPS)*100 ...
             - 60*ev.mu_std - 100*abs(ev.mu0) - 800*ev.x_std - 0.05*ev.pwm_std ...
             - 8*min(ev.recov,3.0);   % B：恢复越快越好
end

function printEval(ev,name)
    fprintf('  %-16s 存活 %6.1f | mu_std %.3f | mu0 %+.3f | PWM_std %6.1f | x_std %.4f | 饱和 %.1f%% | 恢复 %.2fs | 得分 %6.2f\n', ...
        name, ev.steps, ev.mu_std, ev.mu0, ev.pwm_std, ev.x_std, ev.sat, ev.recov, ev.score);
end

function [lA,lC,gA,gC]=ppoGrad(actorNet,criticNet,cloneNet,mS,mA,mL,mT,mAd,eb,cfg)
    oa=forward(actorNet,mS);
    mu_raw=oa(1,:);
    lstd=min(max(oa(2,:)-cfg.LOG_STD_OFFSET,-20),2);
    sig=max(exp(lstd),cfg.SIGMA_MIN);
    v=forward(criticNet,mS);
    lC=0.5*mse(v,mT);
    z=mA; at=tanh(z);
    lp=-0.5*((z-mu_raw).^2)./sig.^2 - log(sig) - 0.5*log(2*pi) - log(1-at.^2+1e-6);
    rho=exp(lp-mL);
    s1=rho.*mAd; s2=max(min(rho,1+cfg.CLIP_EPS),1-cfg.CLIP_EPS).*mAd;
    lA=-mean(min(s1,s2));
    ent=mean(0.5+0.5*log(2*pi)+log(sig));
    lA=lA-eb*ent;
    % ===== 四条经真机验证的结构约束（在【原点】处计算，与真机工作点一致）=====
    w = 0.25 + 0.75*cfg.diff;    % 权重随难度课程渐进，起步25%避免前期跑偏
    hh = 1e-3;
    % 【S3】锚点每次重采样，覆盖真机实际驻留区域；v=omega=0 以避开饱和区
    % （克隆网络在 omega=3 rad/s 处已 tanh 饱和，梯度恒为 0，不能作锚点）
    K   = cfg.NANCH;
    ANC = [ (rand(1,K,'single')-0.5)*cfg.ANCH_X
             zeros(1,K,'single')
            (rand(1,K,'single')-0.5)*cfg.ANCH_A
             zeros(1,K,'single') ];
    ANC(:,1) = 0;    % 【S6】第一列固定为原点，保证原点一定被约束到
    z0 = dlarray(ANC,'CB');
    ex = z0*0; ex(1,:) = hh;
    ea = z0*0; ea(3,:) = hh;
    ew = z0*0; ew(4,:) = hh;
    % 【S4】把期望符号折进来：sx/sa/sw 在方向正确时为【正】，反向时为【负】。
    % S3 用 mean(abs(...))，只管大小不管方向，导致符号翻转的网络照样"达标"。
    dx = (tanh(subsref(forward(actorNet,z0+ex),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ex),substruct('()',{1,':'}))))/(2*hh);
    da = (tanh(subsref(forward(actorNet,z0+ea),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ea),substruct('()',{1,':'}))))/(2*hh);
    dw = (tanh(subsref(forward(actorNet,z0+ew),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ew),substruct('()',{1,':'}))))/(2*hh);
    % 【S6】符号惩罚改【逐点】：先对每锚点 relu 再平均。
    % S5 是 relu(-mean(...))，均值为正可掩盖个别锚点为负 —— 结果原点符号翻转，
    % 断点门槛连续 8080 局拒绝存盘。
    px = cfg.SGN(1)*dx;  pa_ = cfg.SGN(3)*da;  pw = cfg.SGN(4)*dw;
    lA = lA + w*cfg.W_SGN*( mean(relu(-px).^2) + mean(relu(-pa_).^2) + mean(relu(-pw).^2) );
    gx = mean(abs(dx)); ga = mean(abs(da)); gw = mean(abs(dw));

    % 【S8】已删除 Kw/Ka 与 Ka/Kx 两条约束。
    % 理由一（被证伪）：S7 的 Kw/Ka 与克隆完全相同（2.51，跨振幅稳定），
    %   真机主频却是 3.10Hz 而非克隆的 3.96Hz —— 该配比【不决定频率】。
    %   与频率相关的是振幅（0.020->3.96 / 0.038->3.35 / 0.093->3.10Hz）。
    % 理由二（天花板）：这两条本质是"要长得像 LQR"，把 RL 约束成 LQR 的
    %   形状，其性能上限就是 LQR。而真机实测表明改策略形状动不了 Ms，
    %   模仿 LQR 既无必要也无收益。
    % 保留的三条（符号 / 远场推力 / 不饱和）与 LQR 无关，只是控制器的
    % 基本要求，去掉会重蹈 S2（拉不回中心）、S3（推反方向）、S5（bang-bang）。
    % ③【S5】Gamma -> 抖动。改用真机验证过的【时域描述函数】口径：
    %    把 3.95Hz 正弦推过完整观测链路，取输出在该频率的基波幅度。
    %    输入轨迹 cfg.GTD_O 已预计算（不依赖网络），故整条路径可微。
    % 【S8】一次前向算出全部振幅的 Gamma
    oT = tanh(subsref(forward(actorNet,cfg.GTD_O),substruct('()',{1,':'})));
    Lw = cfg.GTD_LW; gA = dlarray(zeros(1,cfg.GTD_NA,'like',extractdata(oT)));
    for ia = 1:cfg.GTD_NA
        seg = oT((ia-1)*Lw+1 : ia*Lw);
        re_ = mean(seg.*cfg.GTD_C); im_ = mean(seg.*cfg.GTD_S);
        gA(ia) = 2*sqrt(re_^2 + im_^2 + 1e-12)/cfg.GTD_AMPS(ia);
    end
    rG = gA ./ cfg.GTD_REF;                     % 逐振幅相对克隆
    % (a) 每个振幅都要达标
    lA = lA + w*cfg.W_GAMMA*sum(relu(rG - cfg.GAMMA_TGT).^2);
    % (b) 禁止随振幅【上升】—— S7 就是 0.71->1.18 一路涨，构成正反馈
    lA = lA + w*cfg.W_GRISE*relu(rG(end) - rG(1))^2;

    % ④ 原点偏置 —— 【必须只在原点算】。若跟随锚点，会强制 mu(x=±0.15)=0，
    %    与"远场要有回复力"直接冲突
    zo = z0(:,1)*0;
    o0 = forward(actorNet, zo);
    lA = lA + w*cfg.W_BIAS*mean(tanh(o0(1,:)).^2);

    % ⑤【S3 新增】远场推力下限 —— 防止"处处偏软"（S2 的真机失败模式）
    PA  = dlarray(single(cfg.AUTH_PTS),'CB');
    mua = tanh(subsref(forward(actorNet,PA),substruct('()',{1,':'})));
    % 无除法形式：要求 mua 与参照【同号】且幅度 >= TGT 倍。
    %   方向正确且够大 -> mua*sign(ref) >= TGT*|ref| -> 违约为 0
    %   方向反了       -> mua*sign(ref) < 0          -> 违约 = TGT*|ref|，重罚
    % 相比除法形式，参照值接近 0 的点不会让比值爆炸。
    proj = mua.*sign(cfg.AUTH_REF);                       % 投影到"正确方向"的分量
    vlo  = relu(cfg.AUTH_LO*abs(cfg.AUTH_REF) - proj);    % 太小（含方向反）
    vhi  = relu(proj - cfg.AUTH_HI*abs(cfg.AUTH_REF));    % 太大（bang-bang）
    lA = lA + w*cfg.W_AUTH*(mean(vlo.^2)+mean(vhi.^2))*16;

    gA=dlgradient(lA,actorNet.Learnables);
    gC=dlgradient(lC,criticNet.Learnables);
end

function m=calcMetrics(net,cfg)
    % 【S3】指标在【锚点】上计算，不再只看原点。
    % S2 的教训：日志显示 Kw/Ka=2.51、Ka/Kx=3.79 全部达标，但那只是原点处的值，
    % 真机工作点（alpha≈0.02）实际是 7.43 / 2.20。原点指标会骗人。
    % 用固定种子，保证不同 iter 之间可比。
    rs=rng(20250903);
    K=cfg.NANCH;
    ANC=[(rand(1,K)-0.5)*cfg.ANCH_X; zeros(1,K); (rand(1,K)-0.5)*cfg.ANCH_A; zeros(1,K)];
    rng(rs);
    h=1e-3; g=zeros(3,K); ci=[1 3 4];
    for q=1:3
        c=ci(q);
        ep=ANC; ep(c,:)=ep(c,:)+h; em=ANC; em(c,:)=em(c,:)-h;
        a=extractdata(predict(net,dlarray(single(ep),'CB')));
        b=extractdata(predict(net,dlarray(single(em),'CB')));
        g(q,:)=abs((tanh(double(a(1,:)))-tanh(double(b(1,:))))/(2*h));
    end
    m.Kx=mean(g(1,:)); m.Ka=mean(g(2,:)); m.Kw=mean(g(3,:));
    m.ratio  = m.Ka/(m.Kx+1e-9);          % Kalpha/Kx  -> 居中（锚点均值）
    m.wratio = m.Kw/(m.Ka+1e-9);          % Komega/Kalpha -> 频率（锚点均值）
    % 【S7】逐点比值的最大偏离——均值达标不代表逐点达标（S6 的教训）
    rwp = g(3,:)./(g(2,:)+1e-9);
    m.wdev = max(abs(rwp/cfg.WRATIO_REF - 1));   % 逐点相对偏离的最大值
    gv = calcGtd(net,cfg); rv = gv./cfg.GTD_REF;
    m.gamma  = max(gv);
    m.gratio = max(rv);                   % 【S8】取全振幅最差
    m.grise  = rv(end) - rv(1);           % >0 表示增益随振幅上升（危险）
    % 【S3 新增】远场推力，相对克隆的比例。最小值 <0.7 即为"偏软"
    oa=extractdata(predict(net,dlarray(single(cfg.AUTH_PTS),'CB')));
    r = tanh(double(oa(1,:)))./cfg.AUTH_REF;             % 带符号比值
    m.auth = min(r); m.authmax = max(r);                 % 【S6】上下都要看
    % 通道符号（正=方向正确）
    h2=1e-3; sg=zeros(1,3); cc=[1 3 4]; sv=[cfg.SGN(1) cfg.SGN(3) cfg.SGN(4)];
    for q=1:3
        sp=zeros(4,1); sm=zeros(4,1); sp(cc(q))=h2; sm(cc(q))=-h2;
        a1=extractdata(predict(net,dlarray(single(sp),'CB')));
        b1=extractdata(predict(net,dlarray(single(sm),'CB')));
        sg(q)=sv(q)*(tanh(double(a1(1)))-tanh(double(b1(1))))/(2*h2);
    end
    m.sgnok = all(sg>0);
end

function g=calcGtd(net,cfg)
    % 【S8】返回逐振幅的 Gamma 向量（1 x nA）
    o = tanh(double(extractdata(predict(net,cfg.GTD_O)))); o = o(1,:);
    Lw = cfg.GTD_LW; g = zeros(1,cfg.GTD_NA);
    for ia = 1:cfg.GTD_NA
        s = o((ia-1)*Lw+1 : ia*Lw);
        re = mean(s.*double(cfg.GTD_C)); im = mean(s.*double(cfg.GTD_S));
        g(ia) = 2*sqrt(re^2+im^2)/cfg.GTD_AMPS(ia);
    end
end

function g=clipGN(g,mx)
    s=0; for i=1:height(g), q=g.Value{i}; s=s+sum(q(:).^2); end
    n=sqrt(extractdata(s));
    if n>mx, g.Value=cellfun(@(q) q*(mx/n), g.Value,'UniformOutput',false); end
end
