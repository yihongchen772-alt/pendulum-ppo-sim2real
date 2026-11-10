%**************************************************************************
% RL 训练 S15 —— S14b 第 640 局续训，把慢回路提上来（2026-11-06；方案和判据见 logs/S15_train/plan.md）
% 相对 S14b 只改训练约束，上机网络不多部件：
%   ① 放开上限：Γ 目标 0.80 → 0.92，推力上限 0.85 → 1.00；
%   ② 加下限：约束锚点平均的 Kv、Kα、Kω 相对起点 +0.7、+0.7、+0.5，前 400 局线性爬到目标。
% 诊断：logs/S14b_train/sway.md。奖励、PPO、低通、T_I、限幅、随机化不变。
%
% 以下是 S14b 的说明：
% RL 训练 S14b —— 相对 S14 只改积分量程：T_I 2 → 0.5 m·s，I_MAX 1 → 4，∂μ/∂I 上限跟着换算（2026-11-06；方案和判据见 logs/S14b_train/plan.md）
% 起点：actor = S14 第 400 局，critic = S14 末断点（I 列除以 4，同一物理积分下 V 不变）；Adam 重开；零偏直接 ±25 mrad；每次评估都存快照。
%
% 以下是 S14 的说明：
% RL 训练 S14 —— S13b + 小车位置积分输入（2026-11-06；方案、地形检查和判据见 logs/S14_train/plan.md）
%
% 目的：不加外环，让网络自己把小车带回中心。只看当前 4 个状态的网络分不清零偏和真实倾斜（内模原理）。
% 相对 S13b 的改动，都服务这一件事：
%   ① 第 5 个输入 I：I += x*DT/T_I，限幅 ±1。积分是写死的，怎么用由网络学（新权重从 0 开始，不预设外环增益）。
%   ② 训练零偏 ±5 mrad 起步，存活够长才加 1 mrad，上限 ±25 mrad。±40 mrad 以上不到 1 s 就越界，学不到东西。
%   ③ 每局 12 → 30 s（积分要时间）；每局随机取 3000 步进更新，所以每次迭代的更新次数和 S13b 相同；冲击频率不变。
%   ④ 约束：∂μ/∂I 限在 [0, GI_MAX]（下限 0 不逼它学，上限防慢环振荡）；导数铰链和 Γ 在 I≠0 处同样要满足。
% 奖励、PPO 超参数、25 Hz 低通、参数随机化都不变。
% 起点：actor = S13b 第 8800 局，critic = S13b 末断点，第一层加一列 0 权重；Adam 重新开始。
%
% 以下是 S13 的说明：
% RL 训练 S13 —— S12 + 网络输出端一阶低通（2026-11-05；计划和判据见 logs/S13_train/plan.md）
%
% 相对 train_rl_S12.m 只改一处：网络输出 u 过一阶低通 y += a(u - y) 再进环境，a = 1 - exp(-2π fc DT)，
% 截止频率 fc 由离线检查 s13_offline.m 按事前规则选定（下面的 FC_HZ）。
% 奖励里的力项和 Δpwm 照旧算在 u 上；探测加在低通之后。奖励、约束、PPO、课程、随机化、种子都和 S12 相同。
% 上机不开外环，用 RodOffset 校零，零偏 ±5 mrad 正好对应校零残差。
%
% 以下是 S12 的说明：
% 与 S8（train_rl_scratch_S8.m，不改动）的唯一实质区别：环境换成校准仿真 v1
% （v1_params / v1_env_init / v1_env_observe / v1_env_step，与 sim_model_v1.py
% 逐拍一致，见 logs/model_v1/matlab_port_check.txt）。
% 奖励、五条约束、PPO 超参数、课程、存盘门槛、得分公式全部照抄 S8。
%
% 环境的变化（S8 -> v1，依据见 logs/model_v1/model_card.md）：
%   转轴阻尼 0.024 -> 实测粘性 0.0006 + 库仑 0.00095；小车阻尼 30 -> 34 + 库仑/静摩擦；
%   推力 pwm/42 -> 1.65*pwm/42，再过 4.6 Hz 一阶滞后和 5.8 ms 纯延迟；
%   角度测量 19.9 ms 延迟 + 真机噪声回放 + 12 位量化（S8：11 位量化、无噪声、无延迟）；
%   过程噪声 2.0 -> 4.08（标定值）；Jp 0.0025 -> 0.00238。
%   观测链（1 Hz / 0.5 Hz 微分器、角速度 x2）与部署代码一致，不再随机化。
% 域随机化：只在 v1 自己不确定的参数附近，宽度随难度 d 放开（d=0 时就是 v1 标称）。
% 零偏：±5 mrad。真机零偏逐日变化（-5 到 -44 mrad）靠外环消掉，网络只需扛住残差。
%
% 实现上的改动（不改算法）：
%   ① 采样时 actor 用手写前向（同一组权重，double 精度），critic 在局末批量算；
%   ② 评估前后保存/恢复随机数状态。S8 的 evaluatePolicy 每局 rng(1000+e)，评估结束后
%      全局随机数落在固定状态，每 10 个 iter 的训练随机序列（初值、冲击时刻、噪声）会重复；
%   ③ 每 10 iter 写断点（网络 + Adam 状态 + 课程 + 随机数），被杀后重跑同一命令即续跑；
%   ④ 评估时额外记录 v1 里的静置抖动和时域 |S|（2.5/3/3.5 Hz，150 PWM，与真机探测同法），
%      只记录、不进门槛和得分；门槛达标时每 50 iter 存一个快照，上机前按物理指标挑。
%
% 用法：matlab -batch "train_rl_S15"；冒烟测试：matlab -batch "SMOKE=1; train_rl_S15"
%**************************************************************************
if ~exist('SMOKE','var'), SMOKE = 0; end
clearvars -except SMOKE; close all;
cd(fileparts(mfilename('fullpath')));

%% ---------------- 校准仿真 v1 ----------------
P = v1_params();
P.MAX_ACTION  = 1000;
P.THETA_LIMIT = 0.2;
P.X_LIMIT     = 0.2;
FC_HZ = 25;                                  % 【S13】输出低通截止频率，见文件头
P.A_LPF = 1 - exp(-2*pi*FC_HZ*P.DT);
Pb = P; Pb.A_LPF = 1;                        % 对照网络（克隆、S8）不加低通

% 域随机化范围（以 v1 标称为中心）
R.Jp   = [0.00214 0.00262];   % 自由摆动实测 0.0024，±10%
R.ka   = [1.40 1.89];         % 混合拟合 1.647、推车 1.640，±15%（供电可能逐日变）
R.fa   = [3.7 5.5];           % 执行器滞后 4.57 Hz，±20%
R.ta   = [0.003 0.009];       % 执行器纯延迟 5.85 ms
R.tal  = [0.015 0.025];       % 角度延迟 19.9 ms（物理来源未定，给宽一点）
R.kf   = [0.60 1.00];         % 摩擦工况系数 0.83
R.B    = [29 39];             % 小车粘性 34，±15%
R.dist = [0.7 1.3];           % 过程噪声倍数
R.nz   = [0.8 1.2];           % 角度噪声倍数
R.BA   = 0.005;               % 零偏残差 ±5 mrad

%% ---------------- 超参数（照抄 S8）----------------
cfg.ACTOR_LR   = 1e-4;
cfg.CRITIC_LR  = 5e-4;
cfg.GAMMA      = 0.99;
cfg.LAMBDA     = 0.95;
cfg.CLIP_EPS   = 0.2;
ENTROPY_BETA   = 5e-3;
K_EPOCHS       = 8;
BATCH_SIZE     = 256;
MAX_EPISODES   = 1600;                        % 【S15】
cfg.MAX_STEPS  = 7500;      % 【S14】30 秒/局
cfg.EVAL_STEPS = 3000;      % 评估仍是 12 秒，得分和旧版可比
cfg.SUB_STEPS  = 3000;      % 【S14】每局随机取这么多步进更新
cfg.KICK_EVERY = 3000;      % 【S14】每 12 秒一个随机时刻冲击一次，频率同 S13b
cfg.T_I        = 0.5;       % 【S14b】积分尺度 m·s：I = ∫x dt / T_I
cfg.I_MAX      = 4.0;       % 【S14b】积分限幅
cfg.GI_MAX     = 0.15;      % 【S14b】∂μ/∂I 上限，等于 g = 0.3 μ/(m·s)，地形检查里 g ≤ 0.4 代价小
cfg.W_GI       = 25.0;      % 【S14】∂μ/∂I 窗口的铰链权重，同 W_SGN
cfg.BA_MAX     = 0.025;     % 【S14b】起点已会回中，零偏直接到顶
cfg.BA_STEP    = 0.001;
cfg.BA_TOP     = 0.025;
cfg.BA_GATE    = 6000;      % 【S14】近 30 局平均存活 ≥ 24 s 才放宽
cfg.cur_theta  = 0.35;
cfg.cur_x      = 0.30;
cfg.diff       = 0.0;
ROLLOUT_EP     = 8;
EVAL_EVERY     = 10;
SNAP_EVERY     = 10;                          % 【S14b】每次评估都存，只影响存盘

cfg.WRATIO_REF = 2.51;      % 只用于日志
cfg.GTD_F      = 3.95;
cfg.GTD_T      = 4.0;
cfg.GTD_AMPS   = [0.010 0.020 0.040 0.060 0.090];
cfg.W_GRISE    = 20.0;
cfg.GAMMA_TGT  = 0.92;      % 【S15】0.80 → 0.92
cfg.W_GAMMA    = 10.0;
cfg.W_BIAS     = 20.0;
cfg.NANCH      = 16;
cfg.ANCH_X     = 0.30;
cfg.ANCH_A     = 0.06;
cfg.AUTH_PTS   = [-0.23 -0.10  0.10  0.23  0.00  0.00  0.00  0.00;
                   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00;
                   0.00  0.00  0.00  0.00  0.10 -0.10  0.04 -0.04;
                   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00;
                   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00];
cfg.AUTH_REF   = [-0.4673 -0.2004 +0.2109 +0.4725 -0.8502 +0.8707 -0.3389 +0.3431];
cfg.AUTH_LO    = 0.45;
cfg.AUTH_HI    = 1.00;      % 【S15】0.85 → 1.00
cfg.AUTH_HI_GATE = 1.40;
cfg.W_AUTH     = 15.0;
cfg.SGN        = [+1, 0, -1, -1];
cfg.W_SGN      = 25.0;
cfg.W_KG       = 100.0;     % 【S15】Kv、Kα、Kω 下限铰链（相对量）
cfg.KG_ADD     = [0.7 0.7 0.5];
cfg.KG_RAMP    = 400;       % 【S15】前 400 局线性爬到目标

cfg.LOG_STD_OFFSET = 1.5;
cfg.SIGMA_MIN      = 0.02;

cfg.W_THETA  = 5.0;
cfg.W_X      = 8.0;
cfg.W_FORCE  = 1.5;
cfg.W_SMOOTH = 6.0;
cfg.R_SURVIVE= 10.0;
cfg.R_FAIL   = -50.0;
cfg.KICK_MAG = 15.0;
ACTOR_GN = 0.5; CRITIC_GN = 5.0;

% 监测用探测（只记录）
cfg.PRB_F    = [2.5 3.0 3.5];
cfg.PRB_SEED = [11 12];
cfg.SW_SEED  = [11 12 13 14];

if SMOKE
    MAX_EPISODES = 24; EVAL_EVERY = 1; SNAP_EVERY = 1;
    OUT = fullfile('logs','S15_smoke');
    NET_OUT = fullfile(OUT,'actor_net_S15.mat'); NET_BEST = fullfile(OUT,'actor_net_S15_best.mat');
else
    OUT = fullfile('logs','S15_train');
    NET_OUT = 'actor_net_S15.mat'; NET_BEST = 'actor_net_S15_best.mat';
end
if ~exist(OUT,'dir'), mkdir(OUT); end
CK = fullfile(OUT,'ckpt.mat');

%% ---------------- 网络 ----------------
S = load('actor_net_lqr_clone.mat');  cloneNet = S.actorNet;   % 只作 Gamma/推力参照，不参与训练
actorLayers = [
    featureInputLayer(5,'Normalization','none','Name','in')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(2,'Name','out')
];
criticLayers = [
    featureInputLayer(5,'Normalization','none','Name','in')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','r1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','r2')
    fullyConnectedLayer(1,'Name','v')
];

% ---- 时域 Gamma 的输入轨迹（照抄 S8；观测链用部署值 0.5 Hz、x2）----
[gb,ga_] = diffFilt(0.5, P.DT);
Ng = round(cfg.GTD_T/P.DT); tg = (0:Ng-1)'*P.DT;
h0 = round(Ng*0.5);
nA = numel(cfg.GTD_AMPS); Lw = Ng-h0+1;
Oall = zeros(4, nA*Lw);
for ia = 1:nA
    thg = cfg.GTD_AMPS(ia)*sin(2*pi*cfg.GTD_F*tg);
    uu=zeros(1,3); yy=zeros(1,3); Og=zeros(4,Ng);
    for n=1:Ng
        alg=-thg(n); uu=[alg uu(1:2)];
        yy=[(gb(1)*uu(1)+gb(2)*uu(2)+gb(3)*uu(3)-ga_(2)*yy(1)-ga_(3)*yy(2)), yy(1:2)];
        Og(:,n)=[0;0;alg;P.OM*yy(1)/10];
    end
    Oall(:,(ia-1)*Lw+1:ia*Lw) = Og(:,h0:end);
end
cfg.GTD_O   = dlarray(single(Oall),'CB');
cfg.GTD_C   = single(cos(2*pi*cfg.GTD_F*tg(h0:end))');
cfg.GTD_S   = single(sin(2*pi*cfg.GTD_F*tg(h0:end))');
cfg.GTD_LW  = Lw; cfg.GTD_NA = nA;
cfg.GTD_REF = calcGtd(cloneNet, cfg);
cfg.GTD_RAW = single(Oall);                     % 【S14】约束里每次更新随机取一个 I
cfg.GTD_O   = dlarray(single([Oall; zeros(1,size(Oall,2))]),'CB');   % 克隆是 4 输入，基准算完再补第 5 行

%% ---------------- 起点 / 断点 ----------------
Sa = load(fullfile('logs','S14b_train','snap_ep00640.mat'));            % 【S15】起点：S14b 第 640 局
cfg.KG0 = anchorGains(Sa.actorNet, cfg);
cfg.KG_T = cfg.KG0 + cfg.KG_ADD;
cfg.KG_MIN = cfg.KG0;
if exist(CK,'file')
    L = load(CK);
    actorNet = L.actorNet; criticNet = L.criticNet;
    avgGA = L.avgGA; avgSGA = L.avgSGA; avgGC = L.avgGC; avgSGC = L.avgSGC; gstep = L.gstep;
    hist_steps = L.hist_steps; total_ep = L.total_ep; iter = L.iter;
    best_score = L.best_score; best_iter = L.best_iter;
    cfg.cur_theta = L.cur_theta; cfg.cur_x = L.cur_x; cfg.diff = L.cur_diff; cfg.BA_MAX = L.cur_ba;
    rng(L.rngs);
    fprintf('\n[%s] 从断点续跑：Ep %d，iter %d，零偏范围 ±%.0f mrad\n', datestr(now,'mm-dd HH:MM'), total_ep, iter, 1000*cfg.BA_MAX);
else
    rng(116);
    Lc = load(fullfile('logs','S14b_train','ckpt.mat'));                % critic、课程来自 S14b 末断点
    actorNet  = Sa.actorNet;
    criticNet = Lc.criticNet;
    avgGA=[]; avgSGA=[]; avgGC=[]; avgSGC=[]; gstep=0;
    hist_steps=[]; total_ep=0; iter=0;
    best_score=-inf; best_iter=0;
    cfg.cur_theta = Lc.cur_theta; cfg.cur_x = Lc.cur_x; cfg.diff = Lc.cur_diff;
    fprintf('【S15】S14b 第 640 局；Γ 目标 %.2f，推力上限 %.2f；Kv/Kα/Kω 起点 %.2f/%.2f/%.2f → 目标 %.2f/%.2f/%.2f；难度 %.2f\n', ...
        cfg.GAMMA_TGT, cfg.AUTH_HI, cfg.KG0, cfg.KG_T, cfg.diff);
end

%% ---------------- PPO ----------------
lossA=dlarray(0); lossC=dlarray(0);
t_start = tic;
fprintf('\n[%s] 开始训练（目标 %d 局）...\n', datestr(now,'mm-dd HH:MM'), MAX_EPISODES);
while total_ep < MAX_EPISODES
    iter = iter+1;
    cfg.KG_MIN = cfg.KG0 + cfg.KG_ADD*min(total_ep/cfg.KG_RAMP, 1);     % 【S15】下限爬坡
    aS=[]; aA=[]; aL=[]; aT=[]; aAdv=[];
    for k=1:ROLLOUT_EP
        if total_ep>=MAX_EPISODES, break; end
        total_ep = total_ep+1;
        R.BA = cfg.BA_MAX;                              % 【S14】当前零偏范围
        [st,ac,lp,adv,td,T] = runEpisode(actorNet, criticNet, P, R, cfg);
        aS=[aS st]; aA=[aA ac]; aL=[aL lp]; aT=[aT td]; aAdv=[aAdv adv]; %#ok<AGROW>
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

    % 课程学习门控（照抄 S8）
    rm = mean(hist_steps(max(1,end-29):end));
    if rm >= cfg.BA_GATE && numel(hist_steps) >= 30      % 【S14】存活够长才放宽零偏
        cfg.BA_MAX = min(cfg.BA_TOP, cfg.BA_MAX + cfg.BA_STEP);
    end
    if rm >= cfg.EVAL_STEPS*0.35                         % 难度门限仍按 12 秒算，同 S13b
        cfg.cur_theta = max(P.THETA_LIMIT, cfg.cur_theta - (0.35-P.THETA_LIMIT)/20);
        cfg.cur_x     = max(P.X_LIMIT,     cfg.cur_x     - (0.30-P.X_LIMIT)/20);
        cfg.diff      = min(1.0, cfg.diff + 0.02);
    elseif rm < cfg.EVAL_STEPS*0.15
        cfg.diff      = max(0.0, cfg.diff - 0.01);
    end

    if mod(iter,EVAL_EVERY)==0
        ev = evaluatePolicy(actorNet, P, cfg, 20, true);
        pm = monitorProbe(actorNet, P, cfg);
        ct = monitorCenter(actorNet, P, cfg);
        kg = anchorGains(actorNet, cfg); xs4 = monitorSway(actorNet, P, cfg);   % 【S15】只记录
        flag='';
        ok_shape = (cfg.diff > 0.9) ...
                && ev.gratio <= 1.00 ...
                && ev.grise  <= 0.30 ...
                && abs(ev.mu0) < 0.03 ...
                && ev.auth   >= cfg.AUTH_LO ...
                && ev.authmax <= cfg.AUTH_HI_GATE ...
                && ev.sgnok;
        if (ok_shape || SMOKE) && ev.score > best_score
            best_score = ev.score; best_iter=iter;
            save(NET_BEST,'actorNet'); flag=' *** 新最佳，已保存';
        end
        if (ok_shape || SMOKE) && mod(iter,SNAP_EVERY)==0
            save(fullfile(OUT,sprintf('snap_ep%05d.mat',total_ep)),'actorNet','ev','pm','ct','kg','xs4');
            flag=[flag ' [快照]']; %#ok<AGROW>
        end
        fprintf(['Ep %5d | 训练存活 %6.1f | 评估存活 %6.1f | 静置 θ %.2f° PWM %5.1f x %+6.1fmm | ' ...
                 '|S| %s | Kv/Kα/Kω %.2f/%.2f/%.2f(下限 %.2f/%.2f/%.2f) | ×0.4 x std %.1f | g %.3f(%.3f~%.3f) | 零偏±%2.0f | -20mrad x %+5.0f std %3.0f 撞%d | Kv %.1f/%.1f | mu0 %+.3f | Gamma %.2fx(升%+.2f) | 推力 %.2f~%.2f | 符号 %d | 难度 %.2f | 得分 %6.2f | %4.0f min%s\n'], ...
            total_ep, rm, ev.steps, pm.th_std, pm.pwm_std, pm.x_mean, sprintf('%.2f/',pm.S), ...
            kg, cfg.KG_MIN, xs4, ev.gi(1), ev.gi(2), ev.gi(3), 1000*cfg.BA_MAX, ct.x_mean, ct.x_std, ct.nfail, ev.kv(1), ev.kv(2), ...
            ev.mu0, ev.gratio, ev.grise, ev.auth, ev.authmax, ev.sgnok, cfg.diff, ev.score, toc(t_start)/60, flag);
        % 断点（先写临时文件再改名，避免写一半被杀）
        cur_theta = cfg.cur_theta; cur_x = cfg.cur_x; cur_diff = cfg.diff; cur_ba = cfg.BA_MAX; rngs = rng; %#ok<NASGU>
        tmp = [CK(1:end-4) '_tmp.mat'];
        save(tmp,'actorNet','criticNet','avgGA','avgSGA','avgGC','avgSGC','gstep', ...
            'hist_steps','total_ep','iter','best_score','best_iter', ...
            'cur_theta','cur_x','cur_diff','cur_ba','rngs');
        movefile(tmp, CK, 'f');
    end
end

save(NET_OUT,'actorNet');
fprintf('\n[%s] 训练结束。最佳出现在 iter %d，得分 %.2f；零偏范围到 ±%.0f mrad\n', ...
    datestr(now,'mm-dd HH:MM'), best_iter, best_score, 1000*cfg.BA_MAX);


%% ======================= 子函数 =======================
function [pe,ds,BA] = sampleParamsV1(P,R,d)
    % 难度 d 控制随机化宽度：d=0 全部取 v1 标称，d=1 全宽（与 S8 的 sampleParams 同一写法）
    r=@(a,n) n + d*((a(1)+(a(2)-a(1))*rand()) - n);
    pe = P;
    pe.Jp = r(R.Jp,P.Jp); pe.ka = r(R.ka,P.ka); pe.fa = r(R.fa,P.fa);
    pe.ta = r(R.ta,P.ta); pe.tal = r(R.tal,P.tal); pe.kf = r(R.kf,P.kf); pe.B = r(R.B,P.B);
    ds = r(R.dist,1.0);
    pe.noise = P.noise * r(R.nz,1.0);
    BA = d*(rand()-0.5)*2*R.BA;
end

function I = stepI(I, x, P, cfg)
    % 【S14】位置积分，上机时用同样的写法
    I = max(min(I + x*P.DT/cfg.T_I, cfg.I_MAX), -cfg.I_MAX);
end

function W = getW(net)
    % 按顺序取（各网络层名不同）：fc1 W/b、fc2 W/b、输出层 W/b，并核对形状
    V = net.Learnables.Value;
    W.W1=undl(V{1}); W.b1=undl(V{2}); W.W2=undl(V{3}); W.b2=undl(V{4}); W.W3=undl(V{5}); W.b3=undl(V{6});
    assert(numel(V)==6 && isequal(size(W.W1),[64 5]) && isequal(size(W.b1),[64 1]) && isequal(size(W.W2),[64 64]) ...
        && isequal(size(W.b2),[64 1]) && isequal(size(W.W3),[2 64]) && isequal(size(W.b3),[2 1]), 'getW: 网络结构不符');
end

function v = undl(v)
    % 从文件读入的网络参数是普通数组，训练中的是 dlarray
    if isa(v,'dlarray'), v = extractdata(v); end
    v = double(v);
end

function oa = actFwd(W,o)
    h = max(W.W1*o + W.b1, 0); h = max(W.W2*h + W.b2, 0); oa = W.W3*h + W.b3;
end

function [b,a]=diffFilt(f0,DT)
    z=0.9; wn=2*pi*f0; K=2/DT;
    b0=wn^2*K; b1=0; b2=-wn^2*K;
    a0=K^2+2*z*wn*K+wn^2; a1=2*(wn^2-K^2); a2=K^2-2*z*wn*K+wn^2;
    b=[b0 b1 b2]/a0; a=[1 a1/a0 a2/a0];
end

function [states,acts,logps,adv,td,T]=runEpisode(actorNet,criticNet,P,R,cfg)
    [pe,ds,BA] = sampleParamsV1(P,R,cfg.diff);
    th0 = (rand()-0.5)*0.06;
    x0  = (rand()-0.5)*0.10;
    E   = v1_env_init(pe, th0, randi(numel(P.noise))-1, BA, x0);
    W   = getW(actorNet);
    MS=cfg.MAX_STEPS;
    states=zeros(5,MS); acts=zeros(1,MS); logps=zeros(1,MS);
    rew=zeros(1,MS); dones=zeros(1,MS);
    prev_pwm=0; steps=0; done=false; y=0; I=0;
    kick_at = randi([200, cfg.KICK_EVERY]);
    while ~done && steps<MS
        steps=steps+1;
        [obs,E]=v1_env_observe(E,pe);
        I=stepI(I,obs(1),P,cfg); obs=[obs;I];          % 【S14】第 5 个输入
        states(:,steps)=obs;
        oa=actFwd(W,obs);
        mu_raw=oa(1);
        lstd=min(max(oa(2)-cfg.LOG_STD_OFFSET,-20),2);
        sig=max(exp(lstd),cfg.SIGMA_MIN);
        z=mu_raw+sig*randn(); a_t=tanh(z);
        logps(steps)=-0.5*((z-mu_raw)^2)/sig^2 - log(sig) - 0.5*log(2*pi) - log(1-a_t^2+1e-6);
        acts(steps)=z;
        pwm=a_t*P.MAX_ACTION;

        kick=0;
        if steps==kick_at
            kick=cfg.diff*(rand()-0.5)*2*cfg.KICK_MAG;
            kick_at=cfg.KICK_EVERY*ceil(steps/cfg.KICK_EVERY)+randi([200, cfg.KICK_EVERY]);
        end
        y=y+P.A_LPF*(pwm-y);                           % 【S13】输出低通（奖励仍算在 pwm 上）
        E=v1_env_step(E,pe,y,cfg.diff*ds,kick);        % 过程噪声随难度
        xn=E.s(1); thn=E.s(3);

        if abs(thn)>cfg.cur_theta || abs(xn)>cfg.cur_x
            r=cfg.R_FAIL; done=true;
        else
            r = cfg.R_SURVIVE ...
              - cfg.W_THETA *(thn/P.THETA_LIMIT)^2 ...
              - cfg.W_X     *(xn /P.X_LIMIT)^2 ...
              - cfg.W_FORCE *(abs(pwm)/P.MAX_ACTION)^2 ...
              - (0.3+0.7*cfg.diff)*cfg.W_SMOOTH*((pwm-prev_pwm)/(2*P.MAX_ACTION))^2*4;
            done=false;
        end
        rew(steps)=r; dones(steps)=done; prev_pwm=pwm;
    end
    T=steps;
    states=states(:,1:T); acts=acts(1:T); logps=logps(1:T);
    rew=rew(1:T); dones=dones(1:T);
    [obsf,~]=v1_env_observe(E,pe);
    obsf=[obsf; stepI(I,obsf(1),P,cfg)];
    v=double(extractdata(predict(criticNet,dlarray(single([states obsf]),'CB'))));
    vals=v(1:T); ve=v;
    adv=zeros(1,T); g=0;
    for t=T:-1:1
        dlt=rew(t)+cfg.GAMMA*ve(t+1)*(1-dones(t))-ve(t);
        g=dlt+cfg.GAMMA*cfg.LAMBDA*(1-dones(t))*g;
        adv(t)=g;
    end
    td=adv+vals;
    if T>cfg.SUB_STEPS                                 % 【S14】GAE 用整局算，更新只取一部分
        ii=sort(randperm(T,cfg.SUB_STEPS));
        states=states(:,ii); acts=acts(ii); logps=logps(ii); adv=adv(ii); td=td(ii);
    end
end

function ev=evaluatePolicy(net,P,cfg,NEP,withKick)
    % 同 S8：固定种子、v1 标称、零偏 0、过程噪声标定值、40% 处冲击；得分公式不变
    if nargin<5, withKick=true; end
    rs = rng;                                  % 【S12】评估不改变训练的随机数流
    W = getW(net);
    steps=zeros(1,NEP); mus=[]; pwms=[]; xstd=zeros(1,NEP); rec=zeros(1,NEP);
    for e=1:NEP
        rng(1000+e);
        th0=(rand()-0.5)*0.06; x0=(rand()-0.5)*0.08;
        E = v1_env_init(P, th0, randi(numel(P.noise))-1, 0.0, x0);
        MS=cfg.EVAL_STEPS; alive=MS;
        mu_l=zeros(1,MS); pw_l=zeros(1,MS); x_l=zeros(1,MS);
        kick_at=round(MS*0.4); recov=MS; y=0; I=0;
        for t=1:MS
            [obs,E]=v1_env_observe(E,P);
            I=stepI(I,obs(1),P,cfg);
            oa=actFwd(W,[obs;I]);
            mu=tanh(oa(1)); pwm=mu*P.MAX_ACTION;
            mu_l(t)=mu; pw_l(t)=pwm; x_l(t)=E.s(1);
            kick=0; if withKick && t==kick_at, kick=cfg.KICK_MAG; end
            y=y+P.A_LPF*(pwm-y);                       % 【S13】输出低通
            E=v1_env_step(E,P,y,1.0,kick);
            if withKick && t>kick_at && recov==MS && abs(E.s(3))<0.02, recov=t-kick_at; end
            if abs(E.s(3))>P.THETA_LIMIT||abs(E.s(1))>P.X_LIMIT, alive=t; break; end
        end
        steps(e)=alive; m=min(500,alive):alive;
        mus=[mus mu_l(m)]; pwms=[pwms pw_l(m)]; xstd(e)=std(x_l(m)); rec(e)=recov; %#ok<AGROW>
    end
    rng(rs);
    o0=actFwd(W,zeros(5,1));
    ev.steps=mean(steps); ev.mu_std=std(mus); ev.mu0=tanh(o0(1));
    ev.pwm_std=std(pwms); ev.x_std=mean(xstd); ev.recov=mean(rec)*P.DT;
    ev.sat=mean(abs(mus)>0.9)*100;
    mm = calcMetrics(net,cfg);
    ev.ratio=mm.ratio; ev.wratio=mm.wratio; ev.gratio=mm.gratio; ev.Kx=mm.Kx; ev.auth=mm.auth; ev.authmax=mm.authmax; ev.sgnok=mm.sgnok; ev.wdev=mm.wdev; ev.grise=mm.grise; ev.gi=mm.gi; ev.kv=mm.kv;
    ev.score = (ev.steps/cfg.EVAL_STEPS)*100 ...
             - 60*ev.mu_std - 100*abs(ev.mu0) - 800*ev.x_std - 0.05*ev.pwm_std ...
             - 8*min(ev.recov,3.0);
end

function pm = monitorProbe(net,P,cfg)
    % 【S12】只记录：v1 标称、零偏 0，静置 20 s（取 5–20 s 算抖动，口径同真机：读数角度）
    % 后接逐频步进正弦 150 PWM、每频 8 s，|S| = 输出 PWM 与注入在该频率的幅值比（去掉每段第 1 s）
    rs = rng; W = getW(net);
    TW = 20; F = cfg.PRB_F; nS = numel(cfg.PRB_SEED);
    Sv = nan(nS,numel(F)); th = nan(1,nS); pw = nan(1,nS); xm = nan(1,nS);
    for q = 1:nS
        rng(cfg.PRB_SEED(q));
        th0 = (rand-0.5)*0.02;
        E = v1_env_init(P, th0, randi(numel(P.noise))-1, 0.0);
        N = round((TW + 8*numel(F))/P.DT);
        U = zeros(1,N); PR = zeros(1,N); AL = zeros(1,N); X = zeros(1,N); ok = true; y = 0; I = 0;
        for k = 1:N
            tt = k*P.DT;
            [o,E,al] = v1_env_observe(E,P);
            p = 0;
            if tt > TW
                sg = floor((tt-TW)/8);
                if sg < numel(F)
                    ts = tt-TW-sg*8; p = 150*min(ts/0.5,1)*sin(2*pi*F(sg+1)*ts);
                end
            end
            I = stepI(I,o(1),P,cfg);
            oa = actFwd(W,[o;I]);
            y = y + P.A_LPF*(tanh(oa(1))*P.MAX_ACTION - y);        % 【S13】输出低通，探测加在它之后
            pwm = max(min(y + p, 1000), -1000);
            E = v1_env_step(E,P,pwm);
            U(k)=pwm; PR(k)=p; AL(k)=al; X(k)=E.s(1);
            if abs(E.s(3))>0.5 || abs(E.s(1))>0.33, ok=false; break; end
        end
        if ~ok, continue; end
        t = (1:N)*P.DT; ms = t>=5 & t<TW;
        th(q) = std(AL(ms))*180/pi; pw(q) = std(U(ms)); xm(q) = mean(X(ms))*1000;
        for i = 1:numel(F)
            m = t>=TW+8*(i-1)+1 & t<TW+8*i; ez = exp(-1j*2*pi*F(i)*t(m));
            Sv(q,i) = abs(sum((U(m)-mean(U(m))).*ez)) / abs(sum(PR(m).*ez));
        end
    end
    rng(rs);
    pm.S = mean(Sv,1,'omitnan'); pm.th_std = mean(th,'omitnan'); pm.pwm_std = mean(pw,'omitnan');
    pm.x_mean = mean(xm,'omitnan'); pm.nfail = sum(isnan(th));
end

function ct = monitorCenter(net,P,cfg)
    % 【S14】只记录：v1 标称、零偏 -20 mrad、60 s，取 40–60 s 的小车均值和 std，种子同探测
    rs = rng; W = getW(net); nS = numel(cfg.PRB_SEED); xm = nan(1,nS); xs = nan(1,nS);
    N = round(60/P.DT);
    for q = 1:nS
        rng(cfg.PRB_SEED(q));
        E = v1_env_init(P, (rand-0.5)*0.02, randi(numel(P.noise))-1, -0.020);
        X = zeros(1,N); y = 0; I = 0; ok = true;
        for k = 1:N
            [o,E] = v1_env_observe(E,P);
            I = stepI(I,o(1),P,cfg);
            oa = actFwd(W,[o;I]);
            y = y + P.A_LPF*(tanh(oa(1))*P.MAX_ACTION - y);
            E = v1_env_step(E,P,y);
            X(k) = E.s(1);
            if abs(E.s(3))>0.5 || abs(E.s(1))>0.33, ok=false; break; end
        end
        if ok, m = round(40/P.DT):N; xm(q) = 1000*mean(X(m)); xs(q) = 1000*std(X(m)); end
    end
    rng(rs);
    ct.x_mean = mean(xm,'omitnan'); ct.x_std = mean(xs,'omitnan'); ct.nfail = sum(isnan(xm));
end

function xs = monitorSway(net,P,cfg)
    % 【S15】只记录：v1 标称、零偏 0、过程噪声 ×0.4、60 s，取 15–60 s 的 x std（mm）
    rs = rng; W = getW(net); nS = numel(cfg.SW_SEED); v = nan(1,nS); N = round(60/P.DT);
    for q = 1:nS
        rng(cfg.SW_SEED(q));
        E = v1_env_init(P, (rand-0.5)*0.02, randi(numel(P.noise))-1, 0);
        X = zeros(1,N); y = 0; I = 0; ok = true;
        for k = 1:N
            [o,E] = v1_env_observe(E,P);
            I = stepI(I,o(1),P,cfg);
            oa = actFwd(W,[o;I]);
            y = y + P.A_LPF*(tanh(oa(1))*P.MAX_ACTION - y);
            E = v1_env_step(E,P,y,0.4);
            X(k) = E.s(1);
            if abs(E.s(3))>0.5 || abs(E.s(1))>0.33, ok=false; break; end
        end
        if ok, v(q) = 1000*std(X(round(15/P.DT):N)); end
    end
    rng(rs); xs = mean(v,'omitnan');
end

function k = anchorGains(net, cfg)
    % 【S15】固定 64 个锚点（分布同约束锚点）上的平均 [Kv Kα Kω]，ω 换算成真值
    rs = rng(20251106); K = 64;
    A = [(rand(1,K)-0.5)*cfg.ANCH_X; zeros(1,K); (rand(1,K)-0.5)*cfg.ANCH_A; zeros(1,K); (rand(1,K)-0.5)*2*cfg.I_MAX];
    rng(rs); h = 1e-3; c = [2 3 4]; s = [1/5 -1 -0.2]; k = zeros(1,3);
    for q = 1:3
        ep = A; ep(c(q),:) = ep(c(q),:) + h; em = A; em(c(q),:) = em(c(q),:) - h;
        a = extractdata(predict(net,dlarray(single(ep),'CB'))); b = extractdata(predict(net,dlarray(single(em),'CB')));
        k(q) = s(q)*mean((tanh(double(a(1,:))) - tanh(double(b(1,:))))/(2*h));
    end
end

function printEval(ev,name)
    fprintf('  %-12s 存活 %6.1f | mu_std %.3f | mu0 %+.3f | PWM_std %6.1f | x_std %.4f | 饱和 %.1f%% | 恢复 %.2fs | 得分 %6.2f\n', ...
        name, ev.steps, ev.mu_std, ev.mu0, ev.pwm_std, ev.x_std, ev.sat, ev.recov, ev.score);
    fprintf('  %-12s Gamma %.2fx(升%+.2f) | 推力 %.2f~%.2f | 符号 %d | Kw/Ka %.2f | Ka/Kx %.2f\n', ...
        '', ev.gratio, ev.grise, ev.auth, ev.authmax, ev.sgnok, ev.wratio, ev.ratio);
end

function printProbe(pm,name)
    fprintf('  %-12s 静置 θ std %.2f° | PWM std %.1f | x 均值 %+.1f mm | |S| @%s Hz = %s（倒 %d 次）\n', ...
        name, pm.th_std, pm.pwm_std, pm.x_mean, '2.5/3/3.5', sprintf('%.2f ',pm.S), pm.nfail);
end

function [lA,lC,gA,gC]=ppoGrad(actorNet,criticNet,cloneNet,mS,mA,mL,mT,mAd,eb,cfg) %#ok<INUSL>
    % 照抄 S8（train_rl_scratch_S8.m 的 ppoGrad），未改
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
    w = 0.25 + 0.75*cfg.diff;
    hh = 1e-3;
    K   = cfg.NANCH;
    ANC = [ (rand(1,K,'single')-0.5)*cfg.ANCH_X
             zeros(1,K,'single')
            (rand(1,K,'single')-0.5)*cfg.ANCH_A
             zeros(1,K,'single')
            (rand(1,K,'single')-0.5)*2*cfg.I_MAX ];     % 【S14】锚点上 I 也随机，约束在 I≠0 处同样成立
    ANC(:,1) = 0;
    z0 = dlarray(ANC,'CB');
    ex = z0*0; ex(1,:) = hh;
    ea = z0*0; ea(3,:) = hh;
    ew = z0*0; ew(4,:) = hh;
    ei = z0*0; ei(5,:) = hh;
    ev_ = z0*0; ev_(2,:) = hh;
    dx = (tanh(subsref(forward(actorNet,z0+ex),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ex),substruct('()',{1,':'}))))/(2*hh);
    da = (tanh(subsref(forward(actorNet,z0+ea),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ea),substruct('()',{1,':'}))))/(2*hh);
    dw = (tanh(subsref(forward(actorNet,z0+ew),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ew),substruct('()',{1,':'}))))/(2*hh);
    di = (tanh(subsref(forward(actorNet,z0+ei),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ei),substruct('()',{1,':'}))))/(2*hh);
    dv = (tanh(subsref(forward(actorNet,z0+ev_),substruct('()',{1,':'}))) ...
        - tanh(subsref(forward(actorNet,z0-ev_),substruct('()',{1,':'}))))/(2*hh);
    px = cfg.SGN(1)*dx;  pa_ = cfg.SGN(3)*da;  pw = cfg.SGN(4)*dw;
    lA = lA + w*cfg.W_SGN*( mean(relu(-px).^2) + mean(relu(-pa_).^2) + mean(relu(-pw).^2) );
    lA = lA + w*cfg.W_GI*( mean(relu(-di).^2) + mean(relu(di-cfg.GI_MAX).^2) );   % 【S14】∂μ/∂I 窗口，0 在窗内，不逼它学
    lA = lA + w*cfg.W_KG*( relu(1 - mean(dv)/5/cfg.KG_MIN(1))^2 + relu(1 + mean(da)/cfg.KG_MIN(2))^2 + relu(1 + 0.2*mean(dw)/cfg.KG_MIN(3))^2 );   % 【S15】慢回路下限

    Iq = (2*rand()-1)*cfg.I_MAX;                       % 【S14】Γ 也在随机 I 下算，每次更新一个值，计算量不变
    OT = dlarray([cfg.GTD_RAW; Iq*ones(1,size(cfg.GTD_RAW,2),'single')],'CB');
    oT = tanh(subsref(forward(actorNet,OT),substruct('()',{1,':'})));
    Lw = cfg.GTD_LW; gG = dlarray(zeros(1,cfg.GTD_NA,'like',extractdata(oT)));
    for ia = 1:cfg.GTD_NA
        seg = oT((ia-1)*Lw+1 : ia*Lw);
        seg = seg - mean(seg);                         % 【S14】I≠0 时输出有偏置，先去掉再算幅值
        re_ = mean(seg.*cfg.GTD_C); im_ = mean(seg.*cfg.GTD_S);
        gG(ia) = 2*sqrt(re_^2 + im_^2 + 1e-12)/cfg.GTD_AMPS(ia);
    end
    rG = gG ./ cfg.GTD_REF;
    lA = lA + w*cfg.W_GAMMA*sum(relu(rG - cfg.GAMMA_TGT).^2);
    lA = lA + w*cfg.W_GRISE*relu(rG(end) - rG(1))^2;

    zo = z0(:,1)*0;
    o0 = forward(actorNet, zo);
    lA = lA + w*cfg.W_BIAS*mean(tanh(o0(1,:)).^2);

    PA  = dlarray(single(cfg.AUTH_PTS),'CB');
    mua = tanh(subsref(forward(actorNet,PA),substruct('()',{1,':'})));
    proj = mua.*sign(cfg.AUTH_REF);
    vlo  = relu(cfg.AUTH_LO*abs(cfg.AUTH_REF) - proj);
    vhi  = relu(proj - cfg.AUTH_HI*abs(cfg.AUTH_REF));
    lA = lA + w*cfg.W_AUTH*(mean(vlo.^2)+mean(vhi.^2))*16;

    gA=dlgradient(lA,actorNet.Learnables);
    gC=dlgradient(lC,criticNet.Learnables);
end

function m=calcMetrics(net,cfg)
    % 照抄 S8
    rs=rng(20250903);
    K=cfg.NANCH;
    ANC=[(rand(1,K)-0.5)*cfg.ANCH_X; zeros(1,K); (rand(1,K)-0.5)*cfg.ANCH_A; zeros(1,K); zeros(1,K)];
    Ii=(rand(1,K)-0.5)*2*cfg.I_MAX;
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
    m.ratio  = m.Ka/(m.Kx+1e-9);
    m.wratio = m.Kw/(m.Ka+1e-9);
    rwp = g(3,:)./(g(2,:)+1e-9);
    m.wdev = max(abs(rwp/cfg.WRATIO_REF - 1));
    gv = calcGtd(net,cfg); rv = gv./cfg.GTD_REF;
    m.gamma  = max(gv);
    m.gratio = max(rv);
    m.grise  = rv(end) - rv(1);
    oa=extractdata(predict(net,dlarray(single(cfg.AUTH_PTS),'CB')));
    r = tanh(double(oa(1,:)))./cfg.AUTH_REF;
    m.auth = min(r); m.authmax = max(r);
    h2=1e-3; sg=zeros(1,3); cc=[1 3 4]; sv=[cfg.SGN(1) cfg.SGN(3) cfg.SGN(4)];
    for q=1:3
        sp=zeros(5,1); sm=zeros(5,1); sp(cc(q))=h2; sm(cc(q))=-h2;
        a1=extractdata(predict(net,dlarray(single(sp),'CB')));
        b1=extractdata(predict(net,dlarray(single(sm),'CB')));
        sg(q)=sv(q)*(tanh(double(a1(1)))-tanh(double(b1(1))))/(2*h2);
    end
    m.sgnok = all(sg>0);
    % 【S14】只记录：g = ∂μ/∂I / T_I，μ/(m·s)；Kv(∓200 mm)
    ep=ANC; ep(5,:)=Ii+h; em=ANC; em(5,:)=Ii-h;
    a=extractdata(predict(net,dlarray(single(ep),'CB'))); b=extractdata(predict(net,dlarray(single(em),'CB')));
    gi=(tanh(double(a(1,:)))-tanh(double(b(1,:))))/(2*h)/cfg.T_I;
    m.gi=[mean(gi) min(gi) max(gi)];
    e0=0.01; Kv=zeros(1,2); xs=[-0.2 0.2];
    for q=1:2
        a=extractdata(predict(net,dlarray(single([xs(q);e0;0;0;0]),'CB')));
        b=extractdata(predict(net,dlarray(single([xs(q);-e0;0;0;0]),'CB')));
        Kv(q)=(tanh(double(a(1)))-tanh(double(b(1))))/(2*e0);
    end
    m.kv=Kv;
end

function g=calcGtd(net,cfg)
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
