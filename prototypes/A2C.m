%**************************************************************************
% A2C (Advantage Actor-Critic) applied to Cart-Pole balancing problem.
% A2C（优势演员-评论家）算法应用于倒立摆平衡问题。也是整个回合更新 不完整
% Using Custom Deep Learning Loop in MATLAB
%**************************************************************************
clc;
clear;
close all;

% --- 1. Hyperparameter Initialization / 超参数初始化 ---
ACTOR_LR = 0.001;        % Learning rate for Actor / Actor 学习率
CRITIC_LR = 0.01;        % Learning rate for Critic / Critic 学习率 (通常比 Actor 稍大)
GAMMA = 0.99;            % Discount factor / 折扣因子
ENTROPY_BETA = 0.05;     % Entropy bonus coefficient / 策略熵系数，鼓励探索
MAX_EPISODES = 10000;    % Maximum training episodes / 最大回合数
MAX_STEPS = 2000;        % Max steps per episode / 每回合最大步数
ACTION_SPACE = [10 5 2 1 -1 -2 -5 -10]; % Discrete forces / 离散推力
NUM_ACTIONS = length(ACTION_SPACE);

% --- 2. Build Networks (Actor & Critic) / 构建双网络 ---

% Actor Network: Outputs Action Probabilities / 输出动作概率
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

% Critic Network: Outputs State Value V(s) / 输出单一的状态价值
criticLayers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(1, 'Name', 'v_out') % Single scalar output / 标量输出
];
criticNet = dlnetwork(criticLayers);

% Adam Optimizer States / Adam 优化器状态变量 (Actor 和 Critic 独立)
avgGradA = []; avgSqGradA = [];
avgGradC = []; avgSqGradC = [];

% --- 3. Data Logging & Plotting Setup / 记录与绘图准备 ---
history_steps = [];      
figure(1);
hLine = plot(NaN, NaN, '-b.'); 
title('A2C Real-time Training Monitor / A2C 实时训练监控'); 
xlabel('Episodes / 回合数'); ylabel('Steps / 存活步数');
grid on;
fprintf('Starting A2C training... / 开始 A2C 训练...\n');

% --- 4. Main Training Loop / 主训练循环 ---
for episode = 1:MAX_EPISODES
    
    theta = 0; thetaDot = 0; x = 0; xDot = 0;
    
    % Pre-allocate trajectory storage / 预分配内存
    states_hist = zeros(4, MAX_STEPS);
    next_states_hist = zeros(4, MAX_STEPS);
    actions_hist = zeros(1, MAX_STEPS);
    rewards_hist = zeros(1, MAX_STEPS);
    dones_hist = zeros(1, MAX_STEPS); % Need to track terminal states / 需要记录终止状态
    
    steps = 0;
    done = false;
    
    % --- Phase A: Collect Trajectory Data / 阶段 A: 收集轨迹 ---
    while ~done && steps < MAX_STEPS
        steps = steps + 1;
        
        currentState = [x; xDot; theta; thetaDot];
        states_hist(:, steps) = currentState;
        
        % Get action probabilities from Actor / 获取概率
        dlState = dlarray(currentState, 'CB');
        actionProbs = extractdata(predict(actorNet, dlState)); 
        
        % Sample action / 采样动作
        actionProbs = max(actionProbs, 1e-6); 
        actionProbs = actionProbs / sum(actionProbs);
        actionIndex = randsample(1:NUM_ACTIONS, 1, true, actionProbs);
        currentAction = ACTION_SPACE(actionIndex);
        actions_hist(steps) = actionIndex;
        
        % Environment step / 环境物理步进
        [thetaNext, thetaDotNext, ~, xNext, xDotNext] = cart_pole_model(currentAction, theta, thetaDot, x, xDot);
        nextState = [xNext; xDotNext; thetaNext; thetaDotNext];
        next_states_hist(:, steps) = nextState;
        
        if abs(thetaNext) > 0.2 || abs(xNext) > 2.4
            r = -10; done = true;
        else
            r = 1; 
        end
        
        rewards_hist(steps) = r;
        dones_hist(steps) = done;
        
        theta = thetaNext; thetaDot = thetaDotNext; x = xNext; xDot = xDotNext;
    end
    
    % Truncate arrays / 截断
    states_hist = states_hist(:, 1:steps);
    next_states_hist = next_states_hist(:, 1:steps);
    actions_hist = actions_hist(1:steps);
    rewards_hist = rewards_hist(1:steps);
    dones_hist = dones_hist(1:steps);
    
    % --- Phase B: Calculate TD Targets & Advantages / 阶段 B: 计算优势函数 ---
    % Convert to dlarray for prediction / 转换为 dlarray 准备预测
    dlStates = dlarray(states_hist, 'CB');
    dlNextStates = dlarray(next_states_hist, 'CB');
    
    % Get V(s) and V(s') from Critic / 预测当前和下一状态价值
    % IMPORTANT: extractdata acts as STOP-GRADIENT here! / 极其重要：extractdata 起到了截断梯度的作用
    v_current = extractdata(predict(criticNet, dlStates));
    v_next = extractdata(predict(criticNet, dlNextStates));
    
    % TD Target: y_t = r_t + gamma * V(s_{t+1}) * (1 - done)
    td_targets = rewards_hist + GAMMA .* v_next .* (1 - dones_hist);
    
    % Advantage: A_t = y_t - V(s_t)
    advantages = td_targets - v_current;
    
    % Standardize advantages (reduces variance further) / 标准化优势，进一步稳定训练
    % advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
    
    % 修改为：增加长度判断
    if length(advantages) > 10
        advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);
    end
    
    % --- Phase C: Gradient Computation & Update / 阶段 C: 梯度计算与网络更新 ---
    % dlTargets = dlarray(td_targets, 'CB');
    % dlAdvantages = dlarray(advantages, 'CB');
    % 
    % % Compute gradients for both networks / 同时计算双网络梯度
    % [lossA, lossC, gradA, gradC] = dlfeval(@a2cGradients, actorNet, criticNet, ...
    %                                        dlStates, actions_hist, dlTargets, dlAdvantages, NUM_ACTIONS, ENTROPY_BETA);
    % 
    % % Update Actor and Critic separately / 分别更新双网络权重
    % [actorNet, avgGradA, avgSqGradA] = adamupdate(actorNet, gradA, avgGradA, avgSqGradA, episode, ACTOR_LR);
    % [criticNet, avgGradC, avgSqGradC] = adamupdate(criticNet, gradC, avgGradC, avgSqGradC, episode, CRITIC_LR);
    % 

    % --- Phase C: Gradient Computation & Update / 阶段 C: 梯度计算与网络更新 ---
    dlTargets = dlarray(td_targets, 'CB');
    dlAdvantages = dlarray(advantages, 'CB');
    
    % 1. 计算梯度
    [lossA, lossC, gradA, gradC] = dlfeval(@a2cGradients, actorNet, criticNet, ...
                                           dlStates, actions_hist, dlTargets, dlAdvantages, NUM_ACTIONS, ENTROPY_BETA);
    
    % 2. 【新增】：强制梯度裁剪 (Gradient Clipping)
    % 这一步就像给控制系统加上饱和限流，极其关键！
    max_grad = 5.0;
    gradA = dlupdate(@(g) min(max(g, -max_grad), max_grad), gradA);
    gradC = dlupdate(@(g) min(max(g, -max_grad), max_grad), gradC);
    
    % 3. 更新权重
    [actorNet, avgGradA, avgSqGradA] = adamupdate(actorNet, gradA, avgGradA, avgSqGradA, episode, ACTOR_LR);
    [criticNet, avgGradC, avgSqGradC] = adamupdate(criticNet, gradC, avgGradC, avgSqGradC, episode, CRITIC_LR);



    
    % --- Logging / 日志打印 ---
    history_steps(end+1) = steps;
    if mod(episode, 10) == 0
        set(hLine, 'XData', 1:episode, 'YData', history_steps);
        drawnow;
        fprintf('Ep %d: Steps = %d, ActorLoss = %.4f, CriticLoss = %.4f\n', ...
            episode, steps, extractdata(lossA), extractdata(lossC));
    end
     
    if mean(history_steps(max(1, end-200):end)) >= MAX_STEPS - 200
        fprintf('\n*** A2C Converged! Balance maintained. ***\n');
        break;
    end
end

% =========================================================================
% Local Function: Compute A2C Losses and Gradients
% 局部函数：计算 A2C 双网络损失与梯度
% =========================================================================
function [lossActor, lossCritic, gradActor, gradCritic] = a2cGradients(actorNet, criticNet, dlStates, actions, tdTargets, advantages, numActions, entropyBeta)
    
    % 1. Forward passes / 前向传播获取参数
    actionProbs = forward(actorNet, dlStates); % [NumActions x T]
    vValues = forward(criticNet, dlStates);    % [1 x T]
    
    % 2. Critic Loss: Mean Squared Error / Critic 损失：均方误差
    % L_c = 0.5 * mean( (TD_Target - V(s))^2 )
    lossCritic = 0.5 * mse(vValues, tdTargets); 
    
    % 3. Actor Loss / Actor 损失
    T = size(dlStates, 2);
    linearIndices = sub2ind([numActions, T], actions, 1:T);
    probsTaken = actionProbs(linearIndices);
    
    % Standard Policy Gradient Loss weighted by Advantage 
    % 使用优势函数加权的策略梯度
    lossActor = -mean(advantages .* log(probsTaken + 1e-8)); 
    
    % 4. Entropy Bonus (Optional but highly recommended for A2C)
    % 策略熵奖励：鼓励 Actor 输出平滑的概率，防止过早陷入局部最优（比如死磕一个推力）
    entropy = -mean(sum(actionProbs .* log(actionProbs + 1e-8), 1));
    lossActor = lossActor - entropyBeta * entropy;
    
    % 5. Compute independent gradients / 分别计算梯度
    gradActor = dlgradient(lossActor, actorNet.Learnables);
    gradCritic = dlgradient(lossCritic, criticNet.Learnables);
end