%**************************************************************************
% SAC (Soft Actor-Critic) for Acrome Inverted Pendulum
%
% 背景：手写的连续PPO（lianxuppov3~v7）反复卡在"训一阵、垮一阵、原地
% 震荡"的平台期，v7换了个思路（去掉不会触发的探索衰减门槛）之后曾经
% 短暂冲到50回合均值~750步，但很快又崩掉了，不够稳定。
%
% 这版换成MATLAB Reinforcement Learning Toolbox官方的rlSACAgent，
% 不再手写PPO的rollout/GAE/梯度裁剪这些容易出bug的训练细节，理由：
% 1. SAC自带温度(entropy weight)自动调节，不需要像PPO那样手动定
%    ENTROPY_BETA怎么衰减——这正是我们在PPO里反复调却没调对的地方。
% 2. SAC是off-policy、带经验回放(replay buffer)，一份数据能反复利用
%    很多次，样本效率比PPO这种on-policy方法高得多，冷启动阶段
%    （"随便怎么动都撑不过150步"那个探索平台期）应该更容易走出来。
% 3. Toolbox内部的tanh-squash高斯策略，log_prob的雅可比修正是官方
%    验证过的，不会有我们手写PPO里"训练记录的概率和实际用到系统上的
%    动作分布对不上"这种潜在偏差。
%
% 物理模型和奖励函数原样复用PPO脚本那一套（见PendulumEnv.m），执行器
% 低通滤波alpha=0.15、观测噪声、reward权重全部不变，这样降低了"到底是
% 环境变了还是算法变了"的混淆变量，也保证训出来的策略跟现有部署代码
% 结构上兼容。
%
% 这版先不加PPO里那套课程学习/表现门控，保持SAC最朴素的标准用法，
% 排除额外变量看SAC本身够不够用；如果冷启动阶段还是卡住，再考虑要不要
% 把课程学习搬过来。
%**************************************************************************
clc; clear; close all;

env = PendulumEnv();

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

rng(0);  % 固定随机种子，方便复现/对比

% 网络隐藏层宽度跟PPO脚本保持一致(64)，方便比较
initOpts = rlAgentInitializationOptions('NumHiddenUnit', 64);

agentOpts = rlSACAgentOptions;
agentOpts.SampleTime            = -1;      % 自定义环境不依赖固定采样时间
agentOpts.DiscountFactor        = 0.99;    % 跟PPO的GAMMA一致
agentOpts.MiniBatchSize         = 256;
agentOpts.ExperienceBufferLength = 2e5;
agentOpts.TargetSmoothFactor    = 5e-3;    % 目标网络软更新系数，官方默认值
agentOpts.ActorOptimizerOptions.LearnRate  = 1e-4;
agentOpts.CriticOptimizerOptions(1).LearnRate = 1e-3;
agentOpts.CriticOptimizerOptions(2).LearnRate = 1e-3;
agentOpts.EntropyWeightOptions.LearnRate   = 1e-4;   % 温度自动调节的学习率
agentOpts.EntropyWeightOptions.TargetEntropy = -1;   % 1维动作，官方默认建议值

agent = rlSACAgent(obsInfo, actInfo, initOpts, agentOpts);

trainOpts = rlTrainingOptions( ...
    'MaxEpisodes', 3000, ...
    'MaxStepsPerEpisode', 5000, ...
    'ScoreAveragingWindowLength', 50, ...
    'Verbose', true, ...
    'Plots', 'none', ...                 % 后台无GUI跑训练，training-progress窗口在-batch下会报错，改成看Verbose文字日志
    'StopTrainingCriteria', 'AverageSteps', ...
    'StopTrainingValue', 4500, ...
    'SaveAgentCriteria', 'EpisodeFrequency', ...  % 长时间后台跑，定期存checkpoint防止中途崩溃/被杀掉时前功尽弃
    'SaveAgentValue', 50, ...
    'SaveAgentDirectory', 'savedAgents_sac');

fprintf('开始SAC训练...\n');
trainingStats = train(agent, env, trainOpts);

save('sac_agent.mat', 'agent', 'trainingStats');
fprintf('训练完成！agent已保存到 sac_agent.mat\n');

%% 测试：带观测噪声，跟部署条件一致
fprintf('\n开始测试（带模拟传感器噪声）...\n');
env.MaxSteps = 2000;
obs = reset(env);
rec_theta = zeros(1,2000); rec_x = zeros(1,2000);
rec_force = zeros(1,2000); rec_a = zeros(1,2000);
test_len = 2000;

for t = 1:2000
    rec_theta(t) = env.State(3);
    rec_x(t)     = env.State(1);

    act = getAction(agent, {obs});
    if iscell(act)      % 不同MATLAB版本getAction返回格式可能是cell或普通数组，两种都兼容
        a = act{1};
    else
        a = act;
    end
    rec_a(t) = a;

    [obs, ~, isDone, ~] = step(env, a);
    rec_force(t) = env.PrevPWM;

    if isDone
        fprintf('测试第%d步结束\n', t);
        test_len = t;
        break;
    end
end

rec_theta = rec_theta(1:test_len);
rec_x     = rec_x(1:test_len);
rec_force = rec_force(1:test_len);
rec_a     = rec_a(1:test_len);

figure('Position',[100,100,800,850]);
subplot(4,1,1);
plot(rec_theta,'b-','LineWidth',1.5); hold on;
yline(0,'k--'); yline(env.THETA_LIMIT,'r:'); yline(-env.THETA_LIMIT,'r:');
title('摆杆角度 θ'); ylabel('rad'); grid on; ylim([-env.THETA_LIMIT*1.25, env.THETA_LIMIT*1.25]);

subplot(4,1,2);
plot(rec_x,'m-','LineWidth',1.5); hold on;
yline(0,'k--'); yline(env.X_LIMIT,'r:'); yline(-env.X_LIMIT,'r:');
title('小车位置 x'); ylabel('m'); grid on; ylim([-env.X_LIMIT*1.25, env.X_LIMIT*1.25]);

subplot(4,1,3);
plot(rec_a,'r-','LineWidth',1);
yline(0,'k--');
title('策略输出（归一化动作）');
ylabel('a'); grid on; ylim([-1.1,1.1]);

subplot(4,1,4);
plot(rec_force,'r-','LineWidth',1.2);
yline(300,'g--'); yline(-300,'g--');
title('控制输出 PWM');
ylabel('PWM'); grid on;
xlabel('步数'); ylim([-env.MAX_ACTION*1.1, env.MAX_ACTION*1.1]);
sgtitle(sprintf('SAC测试结果（存活%d步，带噪声测试）', test_len));
