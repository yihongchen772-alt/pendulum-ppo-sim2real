%**************************************************************************
% LQR 克隆网络（纯监督学习，不做任何 PPO）
%
% 目的：做一个"对照组"，回答一个问题——
%   同样的输入、同样的控制律，换成神经网络来算，能不能在真机上跑动？
%   能  -> 部署链路(观测、单位、codegen、PWM输出)全都是对的，问题在RL学到的策略
%   不能-> 问题在链路本身，跟RL学得好不好无关
%
% 【关键设计】完全对齐 Simulink 里 LQR controller 的实际实现：
%   1. 输入直接用模型 3/4 号端口送来的速度(Transfer Fcn 1Hz / 2nd Order 0.5Hz
%      二阶滤波后的)，**不再自己做差分**——这正是LQR用的信号
%   2. 状态顺序跟项目LQR设计脚本一致：[x, alpha, xdot, alphadot]
%   3. 控制律 u = -K*s (力，牛顿)，再按LQR那条链路换算成PWM：
%        力先限幅 ±35 N (模型里 Saturation)
%        再乘 F/Volt 直流增益 0.6262 V/N
%        再乘 1000/12 = 83.333 PWM/V
%        => 合计 52.18 PWM/N
%   4. PWM 上限用 ±1000（模型里 Saturation2 的真实限幅），不是我们之前用的800
%   5. 无执行器低通(alpha=0.15)——LQR没有这一级，克隆也不能有，纯无记忆映射
%**************************************************************************
clc; clear; close all;

% ---- 项目自带LQR设计脚本算出的真实增益，顺序 [x, alpha, xdot, alphadot] ----
K = [-38.7298, 164.2110, -76.3417, 41.0570];

% ---- LQR 那条链路的力->PWM 换算 ----
FORCE_SAT   = 35;         % 模型里 Saturation 的限幅(牛顿)
F_TO_VOLT   = 0.6262;     % F/Volt 传函直流增益 (V/N)
VOLT_TO_PWM = 1000/12;    % Gain 块
F_TO_PWM    = F_TO_VOLT * VOLT_TO_PWM;   % = 52.18 PWM/N
MAX_ACTION  = 1000;       % Saturation2 的真实限幅

fprintf('力->PWM 换算系数 = %.2f PWM/N\n', F_TO_PWM);
fprintf('目标控制律: pwm = %.2f * clip(-K*s, ±%d N), 再限幅 ±%d\n\n', ...
    F_TO_PWM, FORCE_SAT, MAX_ACTION);

% ---- 采样状态空间。范围取真机实际会遇到的工作区间，边缘留余量 ----
% 【改】分层采样：真机常驻在原点附近的小角度区，均匀采样会让这块样本
% 太稀(5万里只有281个)，导致最该准的地方反而最不准、且原点有残余偏置。
% 这里给每个样本乘一个对数均匀的尺度因子 10^U(-2,0)，使样本量级从满量程
% 一直铺到满量程的1%，原点附近密度提高约两个数量级。
N = 400000;
rng(0);
FULL = [0.25; 0.25; 1.5; 3.0];              % 各通道满量程
scale = 10.^(-2*rand(1,N));                  % 对数均匀 [0.01, 1]
S = (rand(4,N)-0.5)*2 .* FULL .* scale;      % 逐样本缩放

% ---- 目标：完全复刻LQR ----
u_force    = -(K * S);                                   % 牛顿
u_force    = min(max(u_force, -FORCE_SAT), FORCE_SAT);   % ±35 N 限幅
target_pwm = u_force * F_TO_PWM;
target_pwm = min(max(target_pwm, -MAX_ACTION), MAX_ACTION);

% ---- 网络输入的归一化，沿用本项目一贯的约定 [x, v/5, alpha, omega/10] ----
obs = [S(1,:); S(3,:)/5.0; S(2,:); S(4,:)/10.0];

% ---- 网络该输出的 mu_raw，反解 tanh ----
target_action = target_pwm / MAX_ACTION;
target_mu_raw = atanh(min(max(target_action, -0.999), 0.999));

% ---- 网络结构跟之前各版一致，便于对比 ----
layers = [
    featureInputLayer(4, 'Normalization','none','Name','state_in')
    fullyConnectedLayer(64,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(2,'Name','fc_out')
];
actorNet = dlnetwork(layers);

% log_std 通道对部署无影响(部署只取 out(1))，给个固定小值保持结构一致
target_logstd = repmat(log(0.05)+0.3, 1, N);

avgG=[]; avgSqG=[]; step=0;
EPOCHS = 60; BATCH = 512; LR = 3e-3;
fprintf('开始训练(纯监督, %d epochs)...\n', EPOCHS);
for ep = 1:EPOCHS
    idx = randperm(N);
    for si = 1:BATCH:N
        ei = min(si+BATCH-1, N); bi = idx(si:ei);
        mbS  = dlarray(obs(:,bi),'CB');
        mbT1 = dlarray(target_mu_raw(bi),'CB');
        mbT2 = dlarray(target_logstd(bi),'CB');
        [L, G] = dlfeval(@bcGrad, actorNet, mbS, mbT1, mbT2);
        step = step + 1;
        lr_now = LR * 0.5^(floor(ep/20));   % 分段降学习率，末期精修
        [actorNet, avgG, avgSqG] = adamupdate(actorNet, G, avgG, avgSqG, step, lr_now);
    end
    if mod(ep,10)==0
        fprintf('  epoch %2d/%2d  loss=%.6f  lr=%.1e\n', ep, EPOCHS, extractdata(L), lr_now);
    end
end

% =========================================================================
% 验证：网络输出 vs 精确LQR控制律，必须足够接近才有资格上机
% =========================================================================
rng(1);
M = 50000;
T = zeros(4,M);
T(1,:) = (rand(1,M)-0.5)*2*0.20;
T(2,:) = (rand(1,M)-0.5)*2*0.20;
T(3,:) = (rand(1,M)-0.5)*2*1.0;
T(4,:) = (rand(1,M)-0.5)*2*2.0;

uf = min(max(-(K*T), -FORCE_SAT), FORCE_SAT);
pwm_true = min(max(uf*F_TO_PWM, -MAX_ACTION), MAX_ACTION);

obsT = [T(1,:); T(3,:)/5.0; T(2,:); T(4,:)/10.0];
out  = extractdata(predict(actorNet, dlarray(obsT,'CB')));
pwm_net = tanh(double(out(1,:))) * MAX_ACTION;

err = pwm_net - pwm_true;
R2 = 1 - sum(err.^2)/sum((pwm_true-mean(pwm_true)).^2);
fprintf('\n===== 克隆精度验证 (5万个随机状态) =====\n');
fprintf('  PWM 平均绝对误差 : %.2f\n', mean(abs(err)));
fprintf('  PWM 最大绝对误差 : %.2f\n', max(abs(err)));
fprintf('  RMSE             : %.2f\n', sqrt(mean(err.^2)));
fprintf('  R^2              : %.6f\n', R2);

% 小角度区(真机实际工作区)单独看，这里精度最重要
small = abs(T(1,:))<0.05 & abs(T(2,:))<0.05 & abs(T(3,:))<0.3 & abs(T(4,:))<0.6;
fprintf('  小角度区(%d个)平均绝对误差: %.2f  最大: %.2f\n', ...
    sum(small), mean(abs(err(small))), max(abs(err(small))));

% 原点偏置 —— 必须接近0，LQR在原点输出就是0
o0 = extractdata(predict(actorNet, dlarray([0;0;0;0],'CB')));
fprintf('  原点输出 PWM     : %+.2f  (LQR理论值 0)\n', tanh(double(o0(1)))*MAX_ACTION);

% 通道增益 vs LQR理论值
h=1e-3; g=zeros(1,4);
for c=1:4
    sp=zeros(4,1); sm=zeros(4,1); sp(c)=h; sm(c)=-h;
    a=extractdata(predict(actorNet,dlarray(sp,'CB')));
    b=extractdata(predict(actorNet,dlarray(sm,'CB')));
    g(c)=(tanh(double(a(1)))-tanh(double(b(1))))/(2*h)*MAX_ACTION;
end
% 理论: d(pwm)/d(obs) ; obs=[x, xdot/5, alpha, alphadot/10]
gt = [-K(1), -K(3)*5, -K(2), -K(4)*10] * F_TO_PWM;
fprintf('  通道增益 网络     : [%+9.1f %+9.1f %+9.1f %+9.1f]\n', g);
fprintf('  通道增益 LQR理论  : [%+9.1f %+9.1f %+9.1f %+9.1f]\n', gt);
fprintf('  相对误差          : [%+8.2f%% %+8.2f%% %+8.2f%% %+8.2f%%]\n', (g-gt)./abs(gt)*100);

save('actor_net_lqr_clone.mat','actorNet');
fprintf('\n已保存 -> actor_net_lqr_clone.mat\n');

function [loss, grad] = bcGrad(net, S, T1, T2)
    out = forward(net, S);
    loss = mean((out(1,:)-T1).^2,'all') + 0.01*mean((out(2,:)-T2).^2,'all');
    grad = dlgradient(loss, net.Learnables);
end
