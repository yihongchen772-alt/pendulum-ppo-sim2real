clc; clear; close all;

%% 系统参数
Mc  = 0.4719;
Mp  = 0.1679;
Lp  = 0.425/2;   % = 0.2125
Jp  = 0.01;
Beq = 30;
Bp  = 0.024;
g   = 9.81;
Rm  = 2.12;
Kt  = 0.08505;
rmp = 0.013745;

%% 手动写线性化A、B矩阵（比 syms 更可靠）
% 状态向量: [x, alpha, x_dot, alpha_dot]
% 分母
denom = (Mc + Mp)*(Jp + Mp*Lp^2) - (Mp*Lp)^2;

A = [0, 0, 1, 0;
     0, 0, 0, 1;
     0, Mp^2*Lp^2*g/denom,        -Beq*(Jp+Mp*Lp^2)/denom,  Bp*Mp*Lp/denom;
     0, Mp*Lp*g*(Mc+Mp)/denom,    -Beq*Mp*Lp/denom,         -Bp*(Mc+Mp)/denom];

% B矩阵：输入是力F（牛顿）
B_force = [0;
           0;
           (Jp + Mp*Lp^2)/denom;
           Mp*Lp/denom];

fprintf('A矩阵:\n'); disp(A)
fprintf('B矩阵(力输入):\n'); disp(B_force)

% 验证：A(4,2) 应该是正数（不稳定）
fprintf('A(4,2) = %.4f，应为正数（不稳定极点）\n', A(4,2))
fprintf('开环极点: '); disp(eig(A)')

%% LQR设计
% 状态顺序: [x, alpha, x_dot, alpha_dot]
Q = diag([3000, 5000, 1, 1]);
R = 2;

K_force = lqr(A, B_force, Q, R);
fprintf('\nLQR增益K（对力）: '); disp(K_force)

%% 换算到PWM
% F = Kt*12/(Rm*rmp*4095) * PWM
K_convert = Kt * 12 / (Rm * rmp * 4095);
fprintf('PWM换算系数 K_convert = %.6f N/PWM\n', K_convert)

K_pwm = K_force / K_convert;
fprintf('LQR增益K（对PWM）: '); disp(K_pwm)

%% 仿真验证（用你的cart_pole_modelsimulink）
% 初始条件：给一个小扰动
x0 = 0; xDot0 = 0; theta0 = 0.05; thetaDot0 = 0;

N = 2000;
TAU = 0.004;
record_x     = zeros(1,N);
record_theta = zeros(1,N);
record_pwm   = zeros(1,N);

x = x0; xDot = xDot0; theta = theta0; thetaDot = thetaDot0;

for t = 1:N
    record_x(t)     = x;
    record_theta(t) = theta;
    
    % LQR控制律（状态顺序要和K对应）
    state = [x; theta; xDot; thetaDot];
    PWM = -K_pwm * state;
    
    % 饱和限幅
    PWM = max(min(PWM, 4095), -4095);
    record_pwm(t) = PWM;
    
    % 推演下一步
    [thetaNext, thetaDotNext, ~, xNext, xDotNext] = ...
        cart_pole_modelsimulink(PWM, theta, thetaDot, x, xDot);
    
    if abs(thetaNext) > 0.3 || abs(xNext) > 0.5
        fprintf('LQR在第%d步越界\n', t);
        record_x     = record_x(1:t);
        record_theta = record_theta(1:t);
        record_pwm   = record_pwm(1:t);
        break;
    end
    
    x = xNext; xDot = xDotNext;
    theta = thetaNext; thetaDot = thetaDotNext;
end

%% 画图
time = (1:length(record_theta)) * TAU;

figure('Position', [100,100,900,600]);
subplot(3,1,1);
plot(time, record_theta, 'b', 'LineWidth', 1.5);
yline(0,'k--'); yline(0.2,'r:'); yline(-0.2,'r:');
ylabel('theta (rad)'); title('LQR仿真验证'); grid on;

subplot(3,1,2);
plot(time, record_x, 'g', 'LineWidth', 1.5);
yline(0,'k--'); yline(0.5,'r:'); yline(-0.5,'r:');
ylabel('x (m)'); grid on;

subplot(3,1,3);
plot(time, record_pwm, 'r', 'LineWidth', 1.2);
ylabel('PWM'); xlabel('时间 (s)'); grid on;