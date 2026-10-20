%**************************************************************************
% Continuous PPO for Acrome Inverted Pendulum - v8
%
% 背景：v7把"探索力度/学习率必须先跑到1800步均值才衰减"这个从来没
% 成立过的死条件去掉，改成无条件缓慢衰减，跑完后曲线仍然是"冲一波、
% 垮一波、原地震荡"，没有实质改善。v7自己的收尾注释已经预判了下一个
% 嫌疑对象：动作采样和log_prob的设计。
%
% 排查确认的问题：v1~v7里，动作是这样来的：
%     mu = tanh(网络输出)              % 已经squash到(-1,1)
%     action_sampled = mu + sigma*randn()   % 又在squash后的空间里叠加高斯噪声
%     action_clipped = clip(action_sampled, -1, 1)  % 硬裁剪
%     log_prob = 按 N(action_sampled; mu, sigma) 算的
% 问题：
%   1. log_prob是对"裁剪前的action_sampled"算的，但实际用到物理系统、
%      真正决定reward的是"裁剪后的action_clipped"——两者在动作经常需要
%      打到边界附近才能维持平衡的场景下（这正是倒立摆的常态）差异很大，
%      导致PPO更新时用的重要性采样比值rho系统性失真。
%   2. 在mu已经被tanh压缩过的空间里再叠加高斯噪声、又裁剪，这个复合
%      变换不是任何标准分布，log_prob公式（简单高斯pdf）从数学上就是错的，
%      不是"近似"，是"公式对不上"。
% 这跟"不管怎么调训练细节参数（梯度裁剪、LR/entropy衰减节奏、courriculum
% 门槛）曲线形状都一样、冲不破天花板"的现象是吻合的：这些细节参数调的
% 是"怎么用梯度"，而这里错的是"梯度本身算的是错误的目标"。
%
% 相对v7的改动（只改动作采样+log_prob这一处，其余训练细节原样保留，
% 便于判断这一个改动是否是真正原因）：
% 【改】换成标准的squashed-Gaussian（SAC/PPO-continuous通用做法）：
%      网络输出的是未squash的高斯均值mu_raw（无界空间），采样
%      z = mu_raw + sigma*randn() 也在无界空间里做，squash放在采样之后：
%      action = tanh(z)。action天然落在(-1,1)，不再需要额外硬裁剪。
%      log_prob按变量替换公式精确计算：
%      log_prob(action) = log N(z; mu_raw, sigma) - log(1 - tanh(z)^2 + eps)
%      更新时（ppoGradientsContinuous）用同一套公式、拿存下来的z和当前
%      网络参数重新算log_prob，这样rho = exp(new_log_prob - old_log_prob)
%      算的才是同一个动作在新旧策略下概率的真实比值。
%
% 其余（表现门控课程学习、批量采集、整体范数梯度裁剪、无条件LR/entropy/
% SIGMA_MIN衰减、观测噪声注入、执行器滤波、力度惩罚权重等）全部沿用v7，
% 未改动。
%
% 重要：如果这版还是同样的震荡不收敛，说明问题不在手写PPO的实现细节
% 上，而是奖励函数设计或物理模型/课程学习节奏本身的问题，需要往那个
% 方向查（对照同时在跑的lianxusac.m用官方Toolbox的结果，如果SAC能收敛
% 而这版仍不收敛，则可以排除"物理模型/奖励函数"，坐实是手写PPO实现
% 上还有别的bug）。
%**************************************************************************
clc; clear; close all;

cfg.ACTOR_LR     = 1e-4;
cfg.CRITIC_LR    = 1e-3;
cfg.GAMMA        = 0.99;
cfg.LAMBDA       = 0.95;
ENTROPY_BETA     = 0.01;
cfg.CLIP_EPSILON = 0.2;
K_EPOCHS         = 10;
BATCH_SIZE       = 128;
MAX_EPISODES     = 5000;
cfg.MAX_STEPS    = 5000;
cfg.MAX_ACTION   = 800;

% 目标边界（奖励函数 + 课程学习最终收紧到的值，也是部署时的真实约束）
cfg.THETA_LIMIT  = 0.2;
cfg.X_LIMIT      = 0.2;

% 课程学习：episode结束边界的起点（比目标宽）
TERM_THETA_LIMIT_START = 0.35;
TERM_X_LIMIT_START     = 0.30;

% 按表现收紧，不再按固定episode数收紧（沿用v7）
CURRICULUM_N_STEPS        = 15;   % 从起点收紧到目标值，总共分几步
CURRICULUM_PERF_THRESHOLD = 350;  % 当前边界下最近30局平均步数达到这个值才收紧一步
THETA_STEP = (TERM_THETA_LIMIT_START - cfg.THETA_LIMIT) / CURRICULUM_N_STEPS;
X_STEP     = (TERM_X_LIMIT_START     - cfg.X_LIMIT)     / CURRICULUM_N_STEPS;

% 批量采集：每次PPO更新前收集这么多个episode
ROLLOUT_EPISODES_PER_UPDATE = 8;

cfg.ACTUATOR_ALPHA = 0.15;   % 必须跟部署fcn里的alpha保持一致

% 观测噪声标准差（沿用v3~v7，来自真机实测稳态抖动量级）
cfg.NOISE_X     = 0.003;
cfg.NOISE_V     = 0.008;
cfg.NOISE_THETA = 0.008;
cfg.NOISE_OMEGA = 0.15;

% 探索增强相关常量，rollout采样和梯度函数都要用同一套
cfg.LOG_STD_OFFSET = 0.3;
cfg.LOG_STD_MIN    = -20;
cfg.LOG_STD_MAX    = 2;
cfg.SIGMA_MIN      = 0.15;

% 梯度裁剪：按整体L2范数裁剪（沿用v6/v7，v6已验证这不是收敛问题的根因）
ACTOR_GRAD_MAX_NORM  = 1.0;
CRITIC_GRAD_MAX_NORM = 5.0;

actorLayers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(2, 'Name', 'fc_out')
];
actorNet = dlnetwork(actorLayers);

criticLayers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'v_out')
];
criticNet = dlnetwork(criticLayers);

avgGradA = []; avgSqGradA = [];
avgGradC = []; avgSqGradC = [];
global_step = 0;
history_steps = [];

NET_FILE = 'actor_net_v8.mat';   % 【改】跟v7的actor_net.mat分开，避免互相覆盖，方便对照

figure(1);
hLine = plot(NaN, NaN, '-b.'); hold on;
hSmooth = plot(NaN, NaN, 'r-', 'LineWidth', 2);
title('Continuous PPO v8 (squashed-Gaussian动作采样+log-prob雅可比修正)');
xlabel('Episodes'); ylabel('Steps'); grid on;
legend('原始','50回合均值');
fprintf('开始训练...\n');

total_episode = 0;
iter = 0;
lossA = dlarray(0); lossC = dlarray(0);

% 收紧进度是持久状态，逐步逼近目标值（沿用v7）
cur_theta_limit = TERM_THETA_LIMIT_START;
cur_x_limit     = TERM_X_LIMIT_START;

while total_episode < MAX_EPISODES
    iter = iter + 1;

    all_states  = [];
    all_actions = [];
    all_oldlogp = [];
    all_targets = [];
    all_adv     = [];

    for k = 1:ROLLOUT_EPISODES_PER_UPDATE
        if total_episode >= MAX_EPISODES
            break;
        end
        total_episode = total_episode + 1;

        [states_hist, actions_norm_hist, old_log_probs_hist, advantages, td_targets, T] = ...
            runEpisode(actorNet, criticNet, cur_theta_limit, cur_x_limit, cfg);

        all_states  = [all_states,  states_hist];  %#ok<AGROW>
        all_actions = [all_actions, actions_norm_hist]; %#ok<AGROW>
        all_oldlogp = [all_oldlogp, old_log_probs_hist]; %#ok<AGROW>
        all_targets = [all_targets, td_targets]; %#ok<AGROW>
        all_adv     = [all_adv,     advantages]; %#ok<AGROW>

        history_steps(end+1) = T; %#ok<AGROW>
    end

    % 整个batch（多个episode拼在一起）统一做advantage归一化
    Ntotal = numel(all_actions);
    if Ntotal > 5
        all_adv = (all_adv - mean(all_adv)) / (std(all_adv) + 1e-8);
    end

    actual_epochs = K_EPOCHS;
    if Ntotal < 32, actual_epochs = 2; end

    for epoch = 1:actual_epochs
        indices = randperm(Ntotal);
        for si = 1:BATCH_SIZE:Ntotal
            ei = min(si+BATCH_SIZE-1, Ntotal);
            bi = indices(si:ei);

            mb_states      = dlarray(all_states(:, bi), 'CB');
            mb_actions     = dlarray(all_actions(bi), 'CB');   % 【改】现在存的是无界空间的z，不是squash后的动作
            mb_oldLogProbs = dlarray(all_oldlogp(bi), 'CB');
            mb_targets     = dlarray(all_targets(bi), 'CB');
            mb_advantages  = dlarray(all_adv(bi), 'CB');

            global_step = global_step + 1;

            [lossA, lossC, gradA, gradC] = dlfeval(@ppoGradientsContinuous, ...
                actorNet, criticNet, ...
                mb_states, mb_actions, mb_oldLogProbs, mb_targets, mb_advantages, ...
                ENTROPY_BETA, cfg.CLIP_EPSILON, cfg);

            gradA = clipByGlobalNorm(gradA, ACTOR_GRAD_MAX_NORM);
            gradC = clipByGlobalNorm(gradC, CRITIC_GRAD_MAX_NORM);

            [actorNet,  avgGradA, avgSqGradA] = adamupdate(actorNet,  gradA, avgGradA, avgSqGradA, global_step, cfg.ACTOR_LR);
            [criticNet, avgGradC, avgSqGradC] = adamupdate(criticNet, gradC, avgGradC, avgSqGradC, global_step, cfg.CRITIC_LR);
        end
    end

    % 表现门控：当前边界下最近30局平均步数够了，才把边界往目标值收紧一步（沿用v7）
    recent_mean = mean(history_steps(max(1,end-29):end));
    if recent_mean >= CURRICULUM_PERF_THRESHOLD
        cur_theta_limit = max(cfg.THETA_LIMIT, cur_theta_limit - THETA_STEP);
        cur_x_limit     = max(cfg.X_LIMIT,     cur_x_limit     - X_STEP);
    end

    % 无条件缓慢衰减（沿用v7）
    cfg.ACTOR_LR     = max(cfg.ACTOR_LR     * 0.997, 1e-5);
    cfg.CRITIC_LR    = max(cfg.CRITIC_LR    * 0.997, 1e-4);
    ENTROPY_BETA     = max(ENTROPY_BETA     * 0.995, 0.0005);
    cfg.CLIP_EPSILON = max(cfg.CLIP_EPSILON * 0.997, 0.05);
    cfg.SIGMA_MIN    = max(cfg.SIGMA_MIN    * 0.995, 0.02);

    if mod(iter, 6) == 0 || total_episode >= MAX_EPISODES
        save(NET_FILE, 'actorNet');
        fprintf('已保存断点 (episode %d) -> %s\n', total_episode, NET_FILE);
    end

    set(hLine, 'XData', 1:total_episode, 'YData', history_steps);
    if total_episode >= 50
        set(hSmooth, 'XData', 1:total_episode, 'YData', movmean(history_steps,50));
    end
    drawnow;
    n_this_batch = min(ROLLOUT_EPISODES_PER_UPDATE, numel(history_steps));
    batch_mean_steps = mean(history_steps(end-n_this_batch+1:end));
    fprintf('Ep %4d(iter %3d) | 本批平均Steps %6.1f | 边界theta=%.3f x=%.3f | ActorLoss %7.4f | CriticLoss %7.4f\n', ...
        total_episode, iter, batch_mean_steps, ...
        cur_theta_limit, cur_x_limit, extractdata(lossA), extractdata(lossC));

    if mean(history_steps(max(1,end-30):end)) >= cfg.MAX_STEPS-500
        fprintf('\n*** 收敛！***\n');
        break;
    end
end

save(NET_FILE, 'actorNet');
fprintf('训练完成！网络已保存到 %s\n', NET_FILE);

%% 测试：用最终目标边界(THETA_LIMIT/X_LIMIT)，带观测噪声，跟部署条件一致
fprintf('\n开始测试（带模拟传感器噪声，目标边界）...\n');
theta=(rand()-0.5)*0.1; thetaDot=0; x=0; xDot=0;
prev_pwm_test = 0;
rec_theta=zeros(1,2000); rec_x=zeros(1,2000); rec_force=zeros(1,2000); rec_mu=zeros(1,2000);
test_len=2000;

for t = 1:2000
    rec_theta(t)=theta; rec_x(t)=x;

    x_obs     = x        + cfg.NOISE_X     * randn();
    v_obs     = xDot     + cfg.NOISE_V     * randn();
    theta_obs = theta    + cfg.NOISE_THETA * randn();
    omega_obs = thetaDot + cfg.NOISE_OMEGA * randn();
    obs = [x_obs; v_obs/5.0; theta_obs; omega_obs/10.0];

    % 测试用确定性动作 = tanh(mu_raw)，跟v7公式一样（v7的mu本来就是tanh(out_actor(1))）
    out_actor = extractdata(predict(actorNet, dlarray(obs,'CB')));
    mu = tanh(out_actor(1));
    rec_mu(t) = mu;

    raw_pwm_test = mu * cfg.MAX_ACTION;
    filtered_pwm_test = cfg.ACTUATOR_ALPHA*raw_pwm_test + (1-cfg.ACTUATOR_ALPHA)*prev_pwm_test;
    prev_pwm_test = filtered_pwm_test;
    currentAction = filtered_pwm_test;

    rec_force(t) = currentAction;

    [thetaNext,thetaDotNext,~,xNext,xDotNext] = ...
        cart_pole_modelsimulink(currentAction,theta,thetaDot,x,xDot);

    if abs(thetaNext)>cfg.THETA_LIMIT || abs(xNext)>cfg.X_LIMIT
        fprintf('测试第%d步结束\n',t);
        test_len=t; break;
    end
    theta=thetaNext; thetaDot=thetaDotNext; x=xNext; xDot=xDotNext;
end

rec_theta=rec_theta(1:test_len);
rec_x=rec_x(1:test_len);
rec_force=rec_force(1:test_len);
rec_mu=rec_mu(1:test_len);

figure('Position',[100,100,800,850]);
subplot(4,1,1);
plot(rec_theta,'b-','LineWidth',1.5); hold on;
yline(0,'k--'); yline(cfg.THETA_LIMIT,'r:'); yline(-cfg.THETA_LIMIT,'r:');
title('摆杆角度 θ'); ylabel('rad'); grid on; ylim([-cfg.THETA_LIMIT*1.25,cfg.THETA_LIMIT*1.25]);

subplot(4,1,2);
plot(rec_x,'m-','LineWidth',1.5); hold on;
yline(0,'k--'); yline(cfg.X_LIMIT,'r:'); yline(-cfg.X_LIMIT,'r:');
title('小车位置 x'); ylabel('m'); grid on; ylim([-cfg.X_LIMIT*1.25,cfg.X_LIMIT*1.25]);

subplot(4,1,3);
plot(rec_mu,'r-','LineWidth',1);
yline(0,'k--');
title('网络原始输出 mu');
ylabel('mu'); grid on; ylim([-1.1,1.1]);

subplot(4,1,4);
plot(rec_force,'r-','LineWidth',1.2);
yline(300,'g--'); yline(-300,'g--');
title('控制输出 PWM');
ylabel('PWM'); grid on;
xlabel('步数'); ylim([-cfg.MAX_ACTION*1.1,cfg.MAX_ACTION*1.1]);
sgtitle(sprintf('测试结果（存活%d步，带噪声测试）',test_len));


function [states_hist, actions_norm_hist, old_log_probs_hist, advantages, td_targets, T] = ...
        runEpisode(actorNet, criticNet, term_theta_limit, term_x_limit, cfg)
% 跑一整个episode，返回该episode自己的GAE优势和td目标（未跨episode混合）

    theta    = (rand()-0.5) * 0.1;
    thetaDot = 0;
    x        = (rand()-0.5) * 0.15;
    xDot     = 0;
    prev_pwm = 0;

    MAX_STEPS = cfg.MAX_STEPS;
    states_hist        = zeros(4, MAX_STEPS);
    actions_norm_hist  = zeros(1, MAX_STEPS);   % 【改】存无界空间的原始采样z，不是squash后的动作
    old_log_probs_hist = zeros(1, MAX_STEPS);
    rewards_hist       = zeros(1, MAX_STEPS);
    dones_hist         = zeros(1, MAX_STEPS);
    values_hist        = zeros(1, MAX_STEPS);

    steps = 0; done = false;

    while ~done && steps < MAX_STEPS
        steps = steps + 1;

        x_obs     = x        + cfg.NOISE_X     * randn();
        v_obs     = xDot     + cfg.NOISE_V     * randn();
        theta_obs = theta    + cfg.NOISE_THETA * randn();
        omega_obs = thetaDot + cfg.NOISE_OMEGA * randn();

        currentState = [x_obs; v_obs/5.0; theta_obs; omega_obs/10.0];
        states_hist(:, steps) = currentState;
        dlState = dlarray(currentState, 'CB');

        out_actor = extractdata(predict(actorNet, dlState));
        v_current = extractdata(predict(criticNet, dlState));

        % 【改】squashed-Gaussian：mu_raw是无界空间的均值，采样和squash分两步做
        mu_raw  = out_actor(1);
        log_std = min(max(out_actor(2) - cfg.LOG_STD_OFFSET, cfg.LOG_STD_MIN), cfg.LOG_STD_MAX);
        sigma   = max(exp(log_std), cfg.SIGMA_MIN);

        z_sampled = mu_raw + sigma * randn();   % 无界空间里采样
        action    = tanh(z_sampled);            % squash到(-1,1)，天然有界，不再需要硬裁剪

        varr     = sigma^2;
        log_prob = -0.5*((z_sampled-mu_raw)^2)/varr - log(sigma) - 0.5*log(2*pi) ...
                   - log(1 - action^2 + 1e-6);   % tanh变换的雅可比修正，这样log_prob对应的才是真正用到系统上的action

        actions_norm_hist(steps)  = z_sampled;
        old_log_probs_hist(steps) = log_prob;
        values_hist(steps)        = v_current;

        raw_pwm = action * cfg.MAX_ACTION;
        filtered_pwm = cfg.ACTUATOR_ALPHA * raw_pwm + (1 - cfg.ACTUATOR_ALPHA) * prev_pwm;
        prev_pwm = filtered_pwm;
        currentAction = filtered_pwm;

        [thetaNext, thetaDotNext, ~, xNext, xDotNext] = ...
            cart_pole_modelsimulink(currentAction, theta, thetaDot, x, xDot);

        if abs(thetaNext) > term_theta_limit || abs(xNext) > term_x_limit
            r    = -50;
            done = true;
        else
            r_survive = 10.0;
            r_theta   = -5.0 * (thetaNext / cfg.THETA_LIMIT)^2;

            r_x = -5.0 * (xNext / cfg.X_LIMIT)^2;
            if abs(xNext) > cfg.X_LIMIT*0.75
                r_x = r_x - 20.0 * ((abs(xNext)-cfg.X_LIMIT*0.75)/(cfg.X_LIMIT*0.25))^2;
            end

            r_force = -2.5 * (abs(currentAction)/cfg.MAX_ACTION)^2;

            if steps > 1
                prev_action = tanh(actions_norm_hist(steps-1));  % 【改】上一步的z也要squash后才能跟当前action比
                action_diff = abs(action - prev_action);
                r_smooth = -3.0 * (action_diff/2.0)^2;
            else
                r_smooth = 0;
            end

            r    = r_survive + r_theta + r_x + r_force + r_smooth;
            done = false;
        end

        rewards_hist(steps) = r;
        dones_hist(steps)   = done;
        theta = thetaNext; thetaDot = thetaDotNext;
        x     = xNext;     xDot     = xDotNext;
    end

    T = steps;
    states_hist        = states_hist(:, 1:T);
    actions_norm_hist  = actions_norm_hist(1:T);
    rewards_hist       = rewards_hist(1:T);
    dones_hist         = dones_hist(1:T);
    old_log_probs_hist = old_log_probs_hist(1:T);
    values_hist        = values_hist(1:T);

    x_obs_f     = x        + cfg.NOISE_X     * randn();
    v_obs_f     = xDot     + cfg.NOISE_V     * randn();
    theta_obs_f = theta    + cfg.NOISE_THETA * randn();
    omega_obs_f = thetaDot + cfg.NOISE_OMEGA * randn();
    dlNextState = dlarray([x_obs_f; v_obs_f/5.0; theta_obs_f; omega_obs_f/10.0], 'CB');
    v_last          = extractdata(predict(criticNet, dlNextState));
    values_extended = [values_hist, v_last];

    advantages  = zeros(1, T);
    gae_running = 0;
    for t = T:-1:1
        delta         = rewards_hist(t) + cfg.GAMMA*values_extended(t+1)*(1-dones_hist(t)) - values_extended(t);
        gae_running   = delta + cfg.GAMMA*cfg.LAMBDA*(1-dones_hist(t))*gae_running;
        advantages(t) = gae_running;
    end
    td_targets = advantages + values_hist;
end


function [lossActor, lossCritic, gradActor, gradCritic] = ...
        ppoGradientsContinuous(actorNet, criticNet, mbStates, mbActions, ...
        mbOldLogProbs, mbTargets, mbAdvantages, entropyBeta, clipEps, cfg)

    out_actor  = forward(actorNet, mbStates);
    mu_raw     = out_actor(1,:);   % 【改】无界空间均值，不再在这里tanh
    log_std    = min(max(out_actor(2,:) - cfg.LOG_STD_OFFSET, cfg.LOG_STD_MIN), cfg.LOG_STD_MAX);
    sigma      = max(exp(log_std), cfg.SIGMA_MIN);
    vValues    = forward(criticNet, mbStates);
    lossCritic = 0.5 * mse(vValues, mbTargets);

    % 【改】mbActions现在是rollout时存的无界空间原始采样z（不是待优化变量，是常数），
    % 用当前（可能已更新过的）网络参数mu_raw/sigma重新算这个z对应的log_prob，
    % 加上tanh的雅可比修正，这样才是"同一个物理动作在新旧策略下概率的真实比值"
    z          = mbActions;
    actionTanh = tanh(z);
    varr       = sigma.^2;
    log_probs  = -0.5*((z-mu_raw).^2)./varr - log(sigma) - 0.5*log(2*pi) ...
                 - log(1 - actionTanh.^2 + 1e-6);
    rho        = exp(log_probs - mbOldLogProbs);
    surrogate1 = rho .* mbAdvantages;
    rho_clipped= max(min(rho,1+clipEps),1-clipEps);
    surrogate2 = rho_clipped .* mbAdvantages;
    lossActor  = -mean(min(surrogate1, surrogate2));
    entropy    = mean(0.5 + 0.5*log(2*pi) + log(sigma));   % 沿用高斯熵近似，未加tanh修正项（不是本次改动重点）
    lossActor  = lossActor - entropyBeta * entropy;

    gradActor  = dlgradient(lossActor,  actorNet.Learnables);
    gradCritic = dlgradient(lossCritic, criticNet.Learnables);
end


function grads = clipByGlobalNorm(grads, maxNorm)
% 按所有参数梯度拼在一起的整体L2范数裁剪，超过maxNorm才按比例整体缩小，
% 不改变梯度方向；范数不超标时原样返回。
    sumSq = 0;
    for i = 1:height(grads)
        g = grads.Value{i};
        sumSq = sumSq + sum(g(:).^2);
    end
    globalNorm = sqrt(extractdata(sumSq));
    if globalNorm > maxNorm
        scale = maxNorm / globalNorm;
        grads.Value = cellfun(@(g) g*scale, grads.Value, 'UniformOutput', false);
    end
end
