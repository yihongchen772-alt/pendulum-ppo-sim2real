%**************************************************************************
% Standard PPO (Proximal Policy Optimization) for Research
% 面向研究的标准近端策略优化算法
% 
% 核心工业级升级：
% 1. GAE (Generalized Advantage Estimation): 消除 TD Error 的单步近视偏差，让优势计算极其平滑。
% 2. Mini-batch Update: 不再把几千步数据一次性塞进网络，而是打乱后分批训练，引入随机性，防止过拟合。
% 3. PPO-Clip 目标函数: 限制策略更新幅度。
%**************************************************************************
clc; clear; close all;

% =========================================================================
% 1. 超参数 (Standard PPO Hyperparameters)
% =========================================================================
ACTOR_LR     = 3e-4;      % PPO 经典的默认学习率 (3e-4 被称为 RL 界的神秘常数)
CRITIC_LR    = 1e-3;      
GAMMA        = 0.99;      % 折扣因子
LAMBDA       = 0.95;      % 【新增】GAE 的平滑参数
ENTROPY_BETA = 0.01;      % 熵系数
CLIP_EPSILON = 0.2;       % PPO 截断范围
K_EPOCHS     = 10;        % 数据重复利用次数 (标准 PPO 通常在 4~10 之间)
BATCH_SIZE   = 64;        % 【新增】Mini-batch 的大小
MAX_EPISODES = 5000;
MAX_STEPS    = 2000;
ACTION_SPACE = [10 5 2 1 0 -1 -2 -5 -10];
NUM_ACTIONS  = length(ACTION_SPACE);

% =========================================================================
% 2. 构建网络 (保持不变)
% =========================================================================
actorLayers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(64, 'Name', 'fc1') % PPO 通常需要稍微宽一点的网络
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(NUM_ACTIONS, 'Name', 'fc_out')
    softmaxLayer('Name', 'action_probs')
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

% =========================================================================
% 3. 绘图与日志
% =========================================================================
history_steps = [];
figure(1);
hLine = plot(NaN, NaN, '-b.');
title('Standard PPO (Research Version) 训练监控');
xlabel('回合数 Episodes'); ylabel('存活步数 Steps');
grid on;
fprintf('开始 Standard PPO 训练...\n');

% =========================================================================
% 4. 主训练循环
% =========================================================================
for episode = 1:MAX_EPISODES
    % --- 把下面这行原来的代码 ---
    % theta = 0; thetaDot = 0; x = 0; xDot = 0;
    
    % --- 替换成这行带随机扰动的代码 ---
    theta = (rand() - 0.5) * 0.1;  % 给一个 -0.05 到 0.05 rad 的随机初始偏角
    thetaDot = 0; x = 0; xDot = 0;
    
    % ... 后面保持不变 ...
    % 轨迹缓存
    states_hist      = zeros(4, MAX_STEPS);
    actions_hist     = zeros(1, MAX_STEPS);
    rewards_hist     = zeros(1, MAX_STEPS);
    dones_hist       = zeros(1, MAX_STEPS);
    old_probs_hist   = zeros(1, MAX_STEPS);
    values_hist      = zeros(1, MAX_STEPS); % 【新增】记录前向传播时的 V(s)
    
    steps = 0; done = false;
    
    % ---------------------------------------------------------------------
    % 阶段 A: 纯收集数据 (Rollout)
    % ---------------------------------------------------------------------
    while ~done && steps < MAX_STEPS
        steps = steps + 1;
        currentState = [x; xDot; theta; thetaDot];
        states_hist(:, steps) = currentState;
        
        dlState = dlarray(currentState, 'CB');
        % 同时获取 Actor 和 Critic 的输出 (不记录梯度)
        actionProbs = extractdata(predict(actorNet, dlState));
        v_current   = extractdata(predict(criticNet, dlState));
        
        actionProbs = max(actionProbs, 1e-6);
        actionProbs = actionProbs / sum(actionProbs);
        actionIndex = randsample(1:NUM_ACTIONS, 1, true, actionProbs);
        currentAction = ACTION_SPACE(actionIndex);
        
        actions_hist(steps)   = actionIndex;
        old_probs_hist(steps) = actionProbs(actionIndex);
        values_hist(steps)    = v_current; % 存下 V(s) 供计算 GAE 用
        
        [thetaNext, thetaDotNext, ~, xNext, xDotNext] = ...
            cart_pole_model(currentAction, theta, thetaDot, x, xDot);
        
        % ---------------- 替换掉原来的奖励判断逻辑 ----------------
if abs(thetaNext) > 0.2 || abs(xNext) > 2.4
    r = -10; 
    done = true;
else
    % 1. 基础生存奖励
    r_survive = 1.0; 
    
    % 2. 角度惩罚 (越偏离 0 度，扣分越多)
    r_theta = -2.0 * (abs(thetaNext) / 0.2); 
    
    % 3. 位置惩罚 (越偏离轨道中心 0 点，扣分越多，防止它越跑越偏)
    r_x = -0.5 * (abs(xNext) / 2.4); 
    
    % 4. 能量消耗惩罚 (力气用得越大，扣分越多，逼迫它多用 0 挡位)
    % 假设你的 ACTION_SPACE 最大值是 10
    r_force = -0.2 * (abs(currentAction) / 10); 
    
    % 综合奖励
    r = r_survive + r_theta + r_x + r_force;
    done = false;
end
        
        rewards_hist(steps) = r;
        dones_hist(steps)   = done;
        
        theta = thetaNext; thetaDot = thetaDotNext; x = xNext; xDot = xDotNext;
    end
    
    % 截断缓存
    states_hist    = states_hist(:, 1:steps);
    actions_hist   = actions_hist(1:steps);
    rewards_hist   = rewards_hist(1:steps);
    dones_hist     = dones_hist(1:steps);
    old_probs_hist = old_probs_hist(1:steps);
    values_hist    = values_hist(1:steps);
    
    % ---------------------------------------------------------------------
    % 阶段 B: 计算 GAE (广义优势估计) 和 TD Target
    % ---------------------------------------------------------------------
    % 获取最后一个状态的价值 V(s_last)
    dlNextState = dlarray([x; xDot; theta; thetaDot], 'CB');
    v_last = extractdata(predict(criticNet, dlNextState));
    
    % 将 values 数组补齐最后一步，方便计算
    values_extended = [values_hist, v_last]; 
    
    advantages = zeros(1, steps);
    gae_running = 0;
    
    % GAE 核心公式：从后往前反推累计优势
    for t = steps:-1:1
        % 单步 TD 误差: r + γ * V(s') - V(s)
        delta = rewards_hist(t) + GAMMA * values_extended(t+1) * (1 - dones_hist(t)) - values_extended(t);
        % 累加平滑: A_t = delta_t + γ * λ * A_{t+1}
        gae_running = delta + GAMMA * LAMBDA * (1 - dones_hist(t)) * gae_running;
        advantages(t) = gae_running;
    end
    
    % 计算目标值: TD Target = Advantage + V(s)
    td_targets = advantages + values_hist;
    
    % 优势函数标准化 (在全量数据上做)
    if length(advantages) > 5
        advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
    end
    
    % ---------------------------------------------------------------------
    % 阶段 C: PPO 网络更新 (带有 Mini-batch 机制)
    % ---------------------------------------------------------------------
    for epoch = 1:K_EPOCHS
        % 产生打乱的随机索引 (Shuffle)
        indices = randperm(steps);
        
        % 将完整数据切割成多个 Mini-batch
        for start_idx = 1:BATCH_SIZE:steps
            end_idx = min(start_idx + BATCH_SIZE - 1, steps);
            batch_idx = indices(start_idx:end_idx); % 取出一小批数据的索引
            
            % 根据索引提取这一小批数据，转为 dlarray
            mb_states     = dlarray(states_hist(:, batch_idx), 'CB');
            mb_actions    = actions_hist(batch_idx);
            mb_oldProbs   = dlarray(old_probs_hist(batch_idx), 'CB');
            mb_targets    = dlarray(td_targets(batch_idx), 'CB');
            mb_advantages = dlarray(advantages(batch_idx), 'CB');
            
            global_step = global_step + 1;
            
            % 调用相同的 Loss 计算函数，但这次传入的是 Mini-batch
            [lossA, lossC, gradA, gradC] = dlfeval(@ppoGradients, ...
                actorNet, criticNet, ...
                mb_states, mb_actions, mb_oldProbs, mb_targets, mb_advantages, ...
                NUM_ACTIONS, ENTROPY_BETA, CLIP_EPSILON);
            
            % 梯度裁剪防爆炸
            max_grad = 0.5; % PPO 默认梯度裁剪范围通常更小
            gradA = dlupdate(@(g) min(max(g, -max_grad), max_grad), gradA);
            gradC = dlupdate(@(g) min(max(g, -max_grad), max_grad), gradC);
            
            % 更新网络权重
            [actorNet,  avgGradA, avgSqGradA] = adamupdate(actorNet,  gradA, avgGradA, avgSqGradA, global_step, ACTOR_LR);
            [criticNet, avgGradC, avgSqGradC] = adamupdate(criticNet, gradC, avgGradC, avgSqGradC, global_step, CRITIC_LR);
        end
    end
    % ---------------------------------------------------------------------
    % 日志打印
    % ---------------------------------------------------------------------
    history_steps(end+1) = steps;
    if mod(episode, 10) == 0
        set(hLine, 'XData', 1:episode, 'YData', history_steps);
        drawnow;
        fprintf('Ep %4d | Steps %4d | ActorLoss %7.4f | CriticLoss %7.4f\n', ...
            episode, steps, extractdata(lossA), extractdata(lossC));
    end
    
    if mean(history_steps(max(1, end-50):end)) >= MAX_STEPS - 5
        fprintf('\n*** PPO (Research) 训练收敛！完美平衡！ ***\n');
        break;
    end
end

% =========================================================================
% 5. 训练完成后的测试与可视化 (验证 "躺平" 策略)
% =========================================================================
fprintf('\n开始测试训练好的策略并绘制分析图...\n');
% --- 同样替换这里 ---
theta = (rand() - 0.5) * 0.1;  % 测试时也给一个初始偏角，看它能不能救回来！
thetaDot = 0; x = 0; xDot = 0;
test_steps = 2000;
% % 初始化测试环境
% theta = 0; thetaDot = 0; x = 0; xDot = 0;
% test_steps = 2000;
% 
% 记录数组
record_theta = zeros(1, test_steps);
record_force = zeros(1, test_steps);

for t = 1:test_steps
    % 记录当前角度
    record_theta(t) = theta;
    
    % 获取当前状态并转为 dlarray
    currentState = [x; xDot; theta; thetaDot];
    dlState = dlarray(currentState, 'CB');
    
    % 使用训练好的 Actor 网络预测动作概率
    actionProbs = extractdata(predict(actorNet, dlState));
    
    % 测试阶段使用"贪心策略"：直接选择网络认为最好的动作（概率最大的）
    [~, maxIdx] = max(actionProbs); 
    currentAction = ACTION_SPACE(maxIdx);
    
    % 记录当前施加的力
    record_force(t) = currentAction;
    
    % 放入【你的带减号的下垂摆模型】中推演下一步
    % 注意：这里调用的是你代码里的 cart_pole_model
    [thetaNext, thetaDotNext, ~, xNext, xDotNext] = ...
        cart_pole_model(currentAction, theta, thetaDot, x, xDot);
        
    % 如果触发截断条件，就提前结束
    if abs(thetaNext) > 0.2 || abs(xNext) > 2.4
        fprintf('测试在第 %d 步提前结束（越界）。\n', t);
        record_theta = record_theta(1:t);
        record_force = record_force(1:t);
        break;
    end
    
    % 更新状态
    theta = thetaNext; thetaDot = thetaDotNext; x = xNext; xDot = xDotNext;
end

% =========================================================================
% 绘制验证图表
% =========================================================================
figure('Name', '验证 PPO 智能体的"躺平"策略', 'Position', [100, 100, 800, 600]);

% 1. 绘制角度 Theta 变化
subplot(2, 1, 1);
plot(1:length(record_theta), record_theta, 'b-', 'LineWidth', 1.5);
hold on;
yline(0, 'k--', 'LineWidth', 1); % 画出 0 度基准线
yline(0.2, 'r:', 'LineWidth', 1.5); % 画出失败边界线上轨
yline(-0.2, 'r:', 'LineWidth', 1.5); % 画出失败边界线下轨
title('摆杆角度 (\theta) 随时间变化');
xlabel('步数 (Steps)');
ylabel('角度 (rad)');
ylim([-0.25, 0.25]);
grid on;
legend('实际角度', '0度基准', '失败边界(>0.2)', 'Location', 'best');

% 2. 绘制输出推力 Force 变化
subplot(2, 1, 2);
plot(1:length(record_force), record_force, 'r-', 'LineWidth', 1.2);
title('智能体输出动作 (Force/PWM) 随时间变化');
xlabel('步数 (Steps)');
ylabel('输出推力');
ylim([min(ACTION_SPACE)-2, max(ACTION_SPACE)+2]); % 根据你的 ACTION_SPACE 设定
grid on;
legend('网络输出动作', 'Location', 'best');

sgtitle('PPO 行为分析：为什么能拿满分？');
% =========================================================================
% 局部函数：PPO 梯度计算 (与上一版一致，但通常叫 PPO Gradients)
% =========================================================================
function [lossActor, lossCritic, gradActor, gradCritic] = ...
        ppoGradients(actorNet, criticNet, mbStates, mbActions, mbOldProbs, mbTargets, mbAdvantages, numActions, entropyBeta, clipEps)
    
    actionProbs = forward(actorNet, mbStates);
    vValues     = forward(criticNet, mbStates);
    
    % Critic Loss
    lossCritic = 0.5 * mse(vValues, mbTargets);
    
    % Actor Surrogate Loss
    T = size(mbStates, 2);
    linearIndices = sub2ind([numActions, T], mbActions, 1:T);
    newProbsTaken = actionProbs(linearIndices);
    
    rho = newProbsTaken ./ (mbOldProbs + 1e-8);
    surrogate1 = rho .* mbAdvantages;
    rho_clipped = max(min(rho, 1 + clipEps), 1 - clipEps);
    surrogate2  = rho_clipped .* mbAdvantages;
    
    lossActor = -mean(min(surrogate1, surrogate2));
    
    % Entropy Bonus
    entropy   = -mean(sum(actionProbs .* log(actionProbs + 1e-8), 1));
    lossActor = lossActor - entropyBeta * entropy;
    
    gradActor  = dlgradient(lossActor,  actorNet.Learnables);
    gradCritic = dlgradient(lossCritic, criticNet.Learnables);
end