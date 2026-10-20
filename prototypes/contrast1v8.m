%**************************************************************************
% REINFORCE 对比实验：单轨迹 (Batch=1) vs 批处理 (Batch=8)
%**************************************************************************
clc; clear; close all;

% --- 实验参数设定 ---
TOTAL_EPISODES = 3200; % 总游戏局数
batch_to_test = [1, 8]; % 我们要对比的两种 Batch 大小
results = cell(1, length(batch_to_test)); % 用于存储实验结果

% --- 执行对比实验 ---
for i = 1:length(batch_to_test)
    current_B = batch_to_test(i);
    fprintf('\n>>> 正在开始实验：Batch Size = %d\n', current_B);
    results{i} = run_training_session(current_B, TOTAL_EPISODES);
end

% --- 最终绘图对比 ---
figure('Color', 'w', 'Name', '强化学习 Batch 效果对比');
hold on;
colors = {'#D95319', '#0072BD'}; % 橙色 vs 蓝色

for i = 1:length(batch_to_test)
    data = results{i};
    % 画出原始数据的半透明细线（显示噪声）
    plot(data, 'Color', [0.8 0.8 0.8], 'HandleVisibility', 'off'); 
    % 画出滑动平均粗线（显示趋势）
    % 使用 movmean 让我们更容易看出哪种策略更稳
    smoothed_data = movmean(data, 30); 
    plot(smoothed_data, 'Color', colors{i}, 'LineWidth', 2.5, ...
        'DisplayName', ['Batch Size = ', num2str(batch_to_test(i))]);
end

title(['REINFORCE 性能对比 (总局数: ', num2str(TOTAL_EPISODES), ')']);
xlabel('Episode (回合)'); ylabel('Steps (存活步数)');
legend('Location', 'southeast'); grid on;
hold off;

% =========================================================================
% 核心训练逻辑封装函数
% =========================================================================
function history_steps = run_training_session(BATCH_SIZE, TOTAL_EPISODES)
    % 继承你代码中的超参数
    LEARNING_RATE = 0.005; GAMMA = 0.99; MAX_STEPS = 2000;
    ACTION_SPACE = [10 5 2 1 -1 -2 -5 -10]; NUM_ACTIONS = 8;
    
  % --- 修正后的神经网络层定义 ---
    layers = [
        featureInputLayer(4, 'Name', 'state_in')
        fullyConnectedLayer(24, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(24, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(NUM_ACTIONS, 'Name', 'fc_out')
        softmaxLayer('Name', 'action_probs')
    ];
    net = dlnetwork(layers);
    averageGrad = []; averageSqGrad = [];
    
    history_steps = [];
    num_updates = TOTAL_EPISODES / BATCH_SIZE; % 总更新次数
    
    % 训练循环
    for iter = 1:num_updates
        batch_states = []; batch_actions = []; batch_G = []; batch_ep_steps = [];
        
        % 收集一个 Batch 的轨迹
        for b = 1:BATCH_SIZE
            theta = 0; thetaDot = 0; x = 0; xDot = 0;
            ep_states = []; ep_actions = []; ep_rewards = [];
            steps = 0; done = false;
            
            while ~done && steps < MAX_STEPS
                steps = steps + 1;
                currS = [x; xDot; theta; thetaDot];
                ep_states = [ep_states, currS];
                
                % 神经网络决策
                dlS = dlarray(currS, 'CB');
                probs = extractdata(predict(net, dlS));
                idx = randsample(1:NUM_ACTIONS, 1, true, probs);
                
                % 物理交互
                [theta, thetaDot, x, xDot] = cart_pole_physics_internal(ACTION_SPACE(idx), theta, thetaDot, x, xDot);
                
                % 奖励判定
                if abs(theta) > 0.2 || abs(x) > 2.4
                    reward = -100; done = true;
                else
                    reward = 1;
                end
                ep_actions = [ep_actions, idx];
                ep_rewards = [ep_rewards, reward];
            end
            
            % 计算回报 G
            G = zeros(1, steps); running_add = 0;
            for t = steps:-1:1
                running_add = running_add * GAMMA + ep_rewards(t);
                G(t) = running_add;
            end
            
            % 拼接数据
            batch_states = [batch_states, ep_states];
            batch_actions = [batch_actions, ep_actions];
            batch_G = [batch_G, G];
            batch_ep_steps(end+1) = steps;
            history_steps(end+1) = steps; % 记录每一局的真实步数
        end
        
        % 梯度更新
        batch_G = (batch_G - mean(batch_G)) / (std(batch_G) + 1e-8);
        [~, grads] = dlfeval(@modelGradients_internal, net, batch_states, batch_actions, batch_G, NUM_ACTIONS);
        [net, averageGrad, averageSqGrad] = adamupdate(net, grads, averageGrad, averageSqGrad, iter, LEARNING_RATE);
        
        if mod(iter * BATCH_SIZE, 80) == 0
            fprintf('已完成局数: %d, 当前平均步数: %.1f\n', iter * BATCH_SIZE, mean(batch_ep_steps));
        end
    end
end

% --- 辅助函数 (放到文件最末尾) ---
function [loss, gradients] = modelGradients_internal(net, states, actions, G, numActions)
    dlStates = dlarray(states, 'CB');
    actionProbs = forward(net, dlStates);
    numSteps = size(states, 2);
    linearIdx = sub2ind([numActions, numSteps], actions, 1:numSteps);
    probsTaken = actionProbs(linearIdx);
    loss = -sum(G .* log(probsTaken + 1e-8)) / numSteps;
    gradients = dlgradient(loss, net.Learnables);
end

function [thetaNext, thetaDotNext, xNext, xDotNext] = cart_pole_physics_internal(F, theta, thetaDot, x, xDot)
    g = 9.8; Mc = 1.0; Mp = 0.1; L = 0.5; dt = 0.02;
    temp = (F + Mp * L * (thetaDot^2) * sin(theta)) / (Mc + Mp);
    thetaAcc = (g * sin(theta) - cos(theta) * temp) / (L * (4/3 - Mp * (cos(theta)^2) / (Mc + Mp)));
    xAcc = temp - Mp * L * thetaAcc * cos(theta) / (Mc + Mp);
    xNext = x + xDot * dt; xDotNext = xDot + xAcc * dt;
    thetaNext = theta + thetaDot * dt; thetaDotNext = thetaDot + thetaAcc * dt;
end