%**************************************************************************
% N-step A2C — 真正在线更新 + 梯度稳定的工程折中方案
%
% 解决纯 Online A2C 的策略崩溃问题：
%   - 每 N 步收集一个小批量，用平均梯度更新（降低方差）
%   - 不等整回合结束（真正在线，区别于原版批量 A2C）
%   - N=5~20 是实际工程常用范围
%
% 与原版对比：
%   原版：等整回合（最多2000步）→ 更新一次
%   本版：每 N=10 步 → 更新一次（回合内可更新多次）
%   纯Online：每 1 步 → 更新一次（方差太大，会崩溃）
%**************************************************************************
clc; clear; close all;

% =========================================================================
% 1. 超参数
% =========================================================================
ACTOR_LR     = 0.001;
CRITIC_LR    = 0.005;
GAMMA        = 0.99;
ENTROPY_BETA = 0.05;
MAX_EPISODES = 10000;
MAX_STEPS    = 2000;
ACTION_SPACE = [10 5 2 1 -1 -2 -5 -10];
NUM_ACTIONS  = length(ACTION_SPACE);

N_STEPS = 20;   % ← 核心超参数：每收集多少步就更新一次
                %   越小越像 Online（方差大），越大越像批量（更新慢）
                %   推荐范围：5~20

% =========================================================================
% 2. 构建网络（与原版相同）
% =========================================================================
actorLayers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(NUM_ACTIONS, 'Name', 'fc_out')
    softmaxLayer('Name', 'action_probs')
];
actorNet = dlnetwork(actorLayers);

criticLayers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'v_out')
];
criticNet = dlnetwork(criticLayers);

avgGradA = []; avgSqGradA = [];
avgGradC = []; avgSqGradC = [];

% =========================================================================
% 3. N-step 特有：全局更新计数器
%    N-step 模式下，更新次数 = 总步数 / N，用于 Adam 偏差修正
% =========================================================================
update_count = 0;

% =========================================================================
% 4. 绘图
% =========================================================================
history_steps = [];
figure(1);
hLine = plot(NaN, NaN, '-b.');
title(sprintf('N-step A2C (N=%d) 实时训练监控', N_STEPS));
xlabel('回合数 Episodes'); ylabel('存活步数 Steps');
grid on;
fprintf('开始 N-step A2C 训练 (N=%d)...\n', N_STEPS);

% =========================================================================
% 5. 主训练循环
% =========================================================================
for episode = 1:MAX_EPISODES

    % 回合初始化
    theta = 0; thetaDot = 0; x = 0; xDot = 0;
    steps = 0;
    done  = false;

    % N-step 缓冲区（存最近 N 步的数据）
    buf_states   = zeros(4, N_STEPS);
    buf_next     = zeros(4, N_STEPS);
    buf_actions  = zeros(1, N_STEPS);
    buf_rewards  = zeros(1, N_STEPS);
    buf_dones    = zeros(1, N_STEPS);
    buf_ptr      = 0;   % 缓冲区写入指针

    % =============================================================
    % 主步进循环
    % =============================================================
    while ~done && steps < MAX_STEPS
        steps   = steps + 1;
        buf_ptr = buf_ptr + 1;

        % ── 采样动作 ──────────────────────────────────────────────
        currentState = [x; xDot; theta; thetaDot];
        dlState      = dlarray(currentState, 'CB');

        actionProbs = extractdata(predict(actorNet, dlState));
        actionProbs = max(actionProbs, 1e-6);
        actionProbs = actionProbs / sum(actionProbs);
        actionIndex = randsample(1:NUM_ACTIONS, 1, true, actionProbs);
        currentAction = ACTION_SPACE(actionIndex);

        % ── 环境步进 ──────────────────────────────────────────────
        [thetaNext, thetaDotNext, ~, xNext, xDotNext] = ...
            cart_pole_model(currentAction, theta, thetaDot, x, xDot);
        nextState = [xNext; xDotNext; thetaNext; thetaDotNext];

        if abs(thetaNext) > 0.2 || abs(xNext) > 2.4
            r    = -10;
            done = true;
        else
            r    = 1;
        end

        % ── 存入缓冲区 ────────────────────────────────────────────
        buf_states(:, buf_ptr)  = currentState;
        buf_next(:,   buf_ptr)  = nextState;
        buf_actions(buf_ptr)    = actionIndex;
        buf_rewards(buf_ptr)    = r;
        buf_dones(buf_ptr)      = double(done);

        theta    = thetaNext;
        thetaDot = thetaDotNext;
        x        = xNext;
        xDot     = xDotNext;

        % ── 判断是否触发更新 ──────────────────────────────────────
        %
        % 两种情况触发更新：
        %   1. 缓冲区满了（走完 N 步）→ 正常的 N-step 更新
        %   2. 回合结束（done=true）  → 把缓冲区剩余数据清空更新
        %      不能丢弃！回合末尾的失败经验（r=-10）是最重要的学习信号
        %
        should_update = (buf_ptr == N_STEPS) || done;

        if should_update && buf_ptr > 0

            % 取出本批有效数据（buf_ptr 步）
            batch_states   = buf_states(:,  1:buf_ptr);
            batch_next     = buf_next(:,    1:buf_ptr);
            batch_actions  = buf_actions(   1:buf_ptr);
            batch_rewards  = buf_rewards(   1:buf_ptr);
            batch_dones    = buf_dones(     1:buf_ptr);

            % ── 计算 TD 目标和优势（批量，N 步数据）─────────────
            dlBatchStates = dlarray(batch_states, 'CB');
            dlBatchNext   = dlarray(batch_next,   'CB');

            v_current = extractdata(predict(criticNet, dlBatchStates));
            v_next    = extractdata(predict(criticNet, dlBatchNext));

            td_targets = batch_rewards + GAMMA .* v_next .* (1 - batch_dones);
            advantages = td_targets - v_current;

            % N-step 批量足够做标准化（buf_ptr 通常 = N_STEPS = 10）
            % 回合末尾 buf_ptr 可能 < N，加长度保护
            if buf_ptr > 3
                advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
            end

            % ── 计算梯度并更新 ────────────────────────────────────
            dlTargets    = dlarray(td_targets, 'CB');
            dlAdvantages = dlarray(advantages, 'CB');

            update_count = update_count + 1;

            [lossA, lossC, gradA, gradC] = dlfeval(@a2cGradients, ...
                actorNet, criticNet, ...
                dlBatchStates, batch_actions, dlTargets, dlAdvantages, ...
                NUM_ACTIONS, ENTROPY_BETA);

            max_grad = 5.0;
            gradA = dlupdate(@(g) min(max(g, -max_grad), max_grad), gradA);
            gradC = dlupdate(@(g) min(max(g, -max_grad), max_grad), gradC);

            [actorNet,  avgGradA, avgSqGradA] = adamupdate(actorNet,  gradA, avgGradA, avgSqGradA, update_count, ACTOR_LR);
            [criticNet, avgGradC, avgSqGradC] = adamupdate(criticNet, gradC, avgGradC, avgSqGradC, update_count, CRITIC_LR);

            % 清空缓冲区，准备下一个 N-step 窗口
            buf_ptr = 0;
        end
    end
    % ── 回合结束 ──────────────────────────────────────────────────

    % =========================================================
    % 6. 日志与收敛判断
    % =========================================================
    history_steps(end+1) = steps;

    if mod(episode, 10) == 0
        set(hLine, 'XData', 1:episode, 'YData', history_steps);
        drawnow;
        fprintf('Ep %4d | Steps = %4d | Updates = %5d | ActorLoss = %.4f | CriticLoss = %.4f\n', ...
            episode, steps, update_count, extractdata(lossA), extractdata(lossC));
    end

    if mean(history_steps(max(1, end-200):end)) >= MAX_STEPS - 200
        fprintf('\n*** N-step A2C 收敛！***\n');
        break;
    end
end


% =========================================================================
% 局部函数：与原版完全相同（支持批量输入 4×T）
% =========================================================================
function [lossActor, lossCritic, gradActor, gradCritic] = ...
        a2cGradients(actorNet, criticNet, dlStates, actions, tdTargets, advantages, numActions, entropyBeta)

    actionProbs = forward(actorNet,  dlStates);  % [numActions × T]
    vValues     = forward(criticNet, dlStates);  % [1 × T]

    % Critic 损失
    lossCritic = 0.5 * mse(vValues, tdTargets);

    % Actor 损失
    T = size(dlStates, 2);
    linearIndices = sub2ind([numActions, T], actions, 1:T);
    probsTaken    = actionProbs(linearIndices);

    lossActor = -mean(advantages .* log(probsTaken + 1e-8));

    % 熵正则化
    entropy   = -mean(sum(actionProbs .* log(actionProbs + 1e-8), 1));
    lossActor = lossActor - entropyBeta * entropy;

    % 独立梯度
    gradActor  = dlgradient(lossActor,  actorNet.Learnables);
    gradCritic = dlgradient(lossCritic, criticNet.Learnables);
end