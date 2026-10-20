%**************************************************************************
% REINFORCE (Policy Gradient) applied to Cart-Pole balancing problem.
% REINFORCE（策略梯度）算法应用于倒立摆平衡问题。
% Using Custom Deep Learning Loop in MATLAB
% 使用 MATLAB 自定义深度学习训练循环
%**************************************************************************

clc;
clear;
close all;

% --- 1. Hyperparameter Initialization / 超参数初始化 ---
LEARNING_RATE = 0.005;   % Learning rate for Adam optimizer / Adam 优化器的学习率
GAMMA = 0.99;            % Discount factor for future rewards / 未来奖励的折扣因子
MAX_EPISODES = 10000;     % Maximum number of training episodes / 最大训练回合数
MAX_STEPS = 2000;         % Max steps per episode to prevent infinite loops / 每回合最大步数，防止死循环
ACTION_SPACE = [10 5 2 1 -1 -2 -5 -10]; % Discrete forces applied to cart / 施加在小车上的离散力
NUM_ACTIONS = length(ACTION_SPACE);

% --- 2. Build Policy Network / 构建策略神经网络 ---
% Input: 4-dim state (x, x_dot, theta, theta_dot) / 输入：4维状态（位置、速度、角度、角速度）
% Output: Probability distribution over 8 actions / 输出：8个动作的概率分布
layers = [
    featureInputLayer(4, 'Normalization', 'none', 'Name', 'state_in')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(NUM_ACTIONS, 'Name', 'fc_out')
    softmaxLayer('Name', 'action_probs') % Convert logits to probabilities / 将输出转换为概率
];

% Convert to dlnetwork for custom training support / 转换为 dlnetwork 以支持自定义梯度训练
net = dlnetwork(layers);

% Initialize Adam optimizer parameters / 初始化 Adam 优化器状态变量
averageGrad = [];
averageSqGrad = [];

% --- 3. Data Logging & Plotting Setup / 数据记录与绘图准备 ---
history_steps = [];      % Record survival steps per episode / 记录每轮存活步数
figure(1);
hLine = plot(NaN, NaN, '-b.'); 
title('REINFORCE Real-time Training Monitor / 实时训练监控'); 
xlabel('Episodes / 回合数'); ylabel('Steps / 存活步数');
grid on;

fprintf('Starting REINFORCE training... / 开始 REINFORCE 训练...\n');

% --- 4. Main Training Loop (Episode-based) / 主训练循环（基于回合） ---
for episode = 1:MAX_EPISODES
    
    % Initialize physical state (continuous variables) / 初始化物理状态
    theta = 0; thetaDot = 0; x = 0; xDot = 0;
    
    % Storage for current episode's trajectory / 记录当前回合的轨迹
    states_history = zeros(4, MAX_STEPS);
    actions_history = zeros(1, MAX_STEPS);
    rewards_history = zeros(1, MAX_STEPS);
    
    steps = 0;
    done = false;
    
    % --- Phase A: Collect Trajectory Data / 阶段 A: 运行一回合，收集轨迹数据 ---
    while ~done && steps < MAX_STEPS
        steps = steps + 1;
        
        % 1. Get current state vector / 获取当前状态向量
        currentState = [x; xDot; theta; thetaDot];
        states_history(:, steps) = currentState;
        
        % 2. Network forward pass to get action probabilities / 网络前向传播获取动作概率
        dlState = dlarray(currentState, 'CB'); % Format as Batch data / 格式化为 Batch 数据
        actionProbs = predict(net, dlState);
        actionProbs_data = extractdata(actionProbs);
        
        % 3. Sample action based on probability distribution / 根据概率分布采样动作
        % Ensure probabilities sum to 1 to avoid numerical errors / 确保概率和为1，避免数值问题
        actionProbs_data = max(actionProbs_data, 1e-6); 
        actionProbs_data = actionProbs_data / sum(actionProbs_data);
        actionIndex = randsample(1:NUM_ACTIONS, 1, true, actionProbs_data);
        currentAction = ACTION_SPACE(actionIndex);
        actions_history(steps) = actionIndex; % Record index for gradient calculation / 记录索引用于梯度计算
        
        % 4. Environment interaction (physics model) / 环境交互（物理模型更新）
        [thetaNext, thetaDotNext, ~, xNext, xDotNext] = cart_pole_model(currentAction, theta, thetaDot, x, xDot);
        
        % 5. Determine reward and termination / 判定奖励与是否结束
        % Condition: Angle > 0.2 rad or Position > 2.4 / 失败条件：角度偏离过大或位置越界
        if abs(thetaNext) > 0.2 || abs(xNext) > 2.4
            r = -100; % Heavy penalty for failure / 失败的严厉惩罚
            done = true;
        else
            r = 1;    % Reward for staying alive / 存活奖励
        end
        
        rewards_history(steps) = r;
        
        % Update current state / 更新状态
        theta = thetaNext; thetaDot = thetaDotNext; x = xNext; xDot = xDotNext;
    end
    
    % Truncate pre-allocated arrays to actual steps / 截断多余的预分配空间
    states_history = states_history(:, 1:steps);
    actions_history = actions_history(1:steps);
    rewards_history = rewards_history(1:steps);
    
    % --- Phase B: Calculate Cumulative Returns (G_t) / 阶段 B: 计算累积折扣回报 ---
    G = zeros(1, steps);
    running_add = 0;
    for t = steps:-1:1
        running_add = running_add * GAMMA + rewards_history(t);
        G(t) = running_add;
    end
    
    % Baseline Trick: Standardize G_t to stabilize training / 标准化回报，提高训练稳定性
    G = (G - mean(G)) / (std(G) + 1e-8); 
    
    % --- Phase C: Gradient Computation & Network Update / 阶段 C: 梯度计算与更新 ---
    % Evaluate loss and gradients using helper function / 调用辅助函数计算损失和梯度
    [loss, gradients] = dlfeval(@modelGradients, net, states_history, actions_history, G, NUM_ACTIONS);
    
    % Apply Adam optimizer to update network weights / 使用 Adam 更新权重
    [net, averageGrad, averageSqGrad] = adamupdate(net, gradients, ...
        averageGrad, averageSqGrad, episode, LEARNING_RATE);
    
    % --- Logging & Visualization / 记录与可视化 ---
    history_steps(end+1) = steps;
    
    if mod(episode, 10) == 0
        set(hLine, 'XData', 1:episode, 'YData', history_steps);
        drawnow;
        fprintf('Episode %d: Steps = %d, Loss = %.4f\n', episode, steps, extractdata(loss));
    end
    
    % Convergence check / 简单收敛判定
    if mean(history_steps(max(1, end-50):end)) >= MAX_STEPS - 5
        fprintf('\n*** Training Converged! Balance maintained. ***\n');
        break;
    end
end

% --- 5. Final Performance Plot / 绘制性能增长趋势图 ---
figure(2);
window_size = 20;
smoothed_steps = movmean(history_steps, window_size);
plot(history_steps, 'Color', [0.8 0.8 0.8], 'DisplayName', 'Raw Steps'); 
hold on;
plot(smoothed_steps, 'b-', 'LineWidth', 2, 'DisplayName', 'Moving Average');
title('REINFORCE Performance Trend / 性能增长趋势');
xlabel('Episode / 回合数'); ylabel('Steps / 存活步数');
legend show; grid on; hold off;

% =========================================================================
% Local Function: Compute Policy Gradient Loss
% 局部函数：计算策略梯度损失
% =========================================================================
function [loss, gradients] = modelGradients(net, states, actions, G, numActions)
    % states: 4 x T (States at each time step / 每个时间步的状态)
    % actions: 1 x T (Indices of actions taken / 所选动作的索引)
    % G: 1 x T (Standardized cumulative returns / 标准化后的累积回报)
    
    % Convert inputs to dlarray / 转换为深度学习专用数组
    dlStates = dlarray(states, 'CB');
    
    % Forward pass to get probabilities for all actions (numActions x T)
    % 前向传播得到所有动作的概率矩阵
    actionProbs = forward(net, dlStates);
    
    % Extract the probabilities of the specific actions that were actually taken
    % 提取轨迹中实际执行的那个动作对应的概率
    T = size(states, 2);
    linearIndices = sub2ind([numActions, T], actions, 1:T);
    probsTaken = actionProbs(linearIndices);
    
    % Policy Gradient Loss Formula: Loss = - sum( G_t * log(P(a_t | s_t)) )
    % 策略梯度损失公式。加 1e-8 防止 log(0) 产生 NaN。
    %  use negative sign because we minimize loss to maximize expected reward.
    % 因为要最大化回报，所以在优化器的最小化过程中使用负号。
    loss = -sum(G .* log(probsTaken + 1e-8)); 
    
    % Automatic differentiation / 自动求导计算梯度
    gradients = dlgradient(loss, net.Learnables);
end