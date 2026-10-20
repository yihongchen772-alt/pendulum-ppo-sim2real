%**************************************************************************
% 上机前体检 —— S8（已验证 207s）vs S10（待测）
%
% 四项检查（历史上各拦下过一个版本）：
%   ① 位置回复力剖面      单调 + 符号正确        拦下过 S2、S3
%   ② 带符号推力（8 点）  落在 [0.45, 1.40]      拦下过 S5（bang-bang 4.50）
%   ③ Gamma 跨振幅        <=1.00 且不随振幅上升   拦下过 S7（1.18，升 +0.38）
%   ④ 静态平衡点          解 mu(x,0,BA,0)=0
%   ⑥ 角速度通道增益      Kw<=32 且 Kw*w_rms/10<0.3   拦下过 S10（2.59 bang-bang）
% 另加：原点雅可比结构、饱和占比
%
% 注意 ② 和 ③ 都是【相对克隆的比值】，对非线性网络门限偏严（S8 曾报 1.90
% 却在真机跑了 207s），且比值会约掉绝对量级 —— 这正是 S10 漏检的原因。
% 故 ⑥ 用真机实测的 omega 幅度算绝对值，不做比值。
%**************************************************************************
clc;
DT = 0.004;  BA = -0.0101;  OM = 2.0;  MAXA = 1000;

nets = {};
cand = {'克隆','actor_net_lqr_clone.mat'
        'S8  ','actor_net_S8_deploy.mat'
        'S10 ','actor_net_S10_deploy.mat'
        '混合AC','actor_net_blendAC.mat'};
for i = 1:size(cand,1)
    if isfile(cand{i,2}), nets = [nets; cand(i,:)]; end
end
NN = size(nets,1);
NET = cell(NN,1);
for i = 1:NN, S = load(nets{i,2}); NET{i} = S.actorNet; end
fprintf('参与体检: %d 个网络\n', NN);

mu = @(net,s) tanh(double(subsref(extractdata(predict(net, ...
        dlarray(single(s),'CB'))), substruct('()',{1,':'}))));

%% ===== ① 位置回复力剖面（alpha=零偏，其余为零）=====
fprintf('\n===== ① 位置回复力剖面  pwm vs x（alpha=%.4f rad）=====\n', BA);
xs = [-200 -150 -100 -50 -20 0 20 50 100 150 200]/1000;
fprintf('%-7s','x(mm)'); fprintf('%7.0f', xs*1000); fprintf('   单调\n');
for i = 1:NN
    p = zeros(1,numel(xs));
    for k = 1:numel(xs), p(k) = mu(NET{i}, [xs(k);0;BA;0])*MAXA; end
    d = diff(p);
    mono = all(d < 0) || all(d > 0);
    fprintf('%-7s', nets{i,1}); fprintf('%7.0f', p);
    fprintf('   %s\n', ternary(mono,'OK','**违规**'));
end

%% ===== ② 带符号推力（8 点，相对克隆）=====
fprintf('\n===== ② 带符号推力（8 个状态点，相对克隆网络）=====\n');
P = [ -0.15 0 0.05 0 ; -0.15 0 -0.05 0 ; 0.15 0 0.05 0 ; 0.15 0 -0.05 0
      -0.05 0 0.10 0 ;  0.05 0 -0.10 0 ; 0 0.3 0.05 1.0 ; 0 -0.3 -0.05 -1.0 ]';
% 先算各网络在 8 点的绝对 PWM
PW = zeros(NN,8);
for i = 1:NN
    for k = 1:8
        PW(i,k) = mu(NET{i}, [P(1,k); P(2,k)/5; P(3,k); P(4,k)/10]) * MAXA;
    end
end
fprintf('绝对 PWM（比值在参考量接近零时无意义，故先看绝对值）:\n');
fprintf('%-7s', 'net'); fprintf('%8d', 1:8); fprintf('\n');
for i = 1:NN
    fprintf('%-7s', nets{i,1}); fprintf('%8.0f', PW(i,:)); fprintf('\n');
end
% 比值：仅在克隆的参考量 >=80 PWM 的点上判定（否则处于零点附近）
BIG = abs(PW(1,:)) >= 80;
fprintf('\n带符号比值（* = 参考量 <80 PWM，处于零点附近，不参与判定）:\n');
fprintf('%-7s', 'net');
for k = 1:8, fprintf('%7d%s', k, ternary(BIG(k),' ','*')); end
fprintf('%9s%9s%9s\n','min','max','判定');
for i = 1:NN
    r  = PW(i,:) ./ PW(1,:);
    rj = r(BIG);
    ok = all(rj >= 0.45) && all(rj <= 1.40);
    fprintf('%-7s', nets{i,1});
    for k = 1:8, fprintf('%7.2f%s', r(k), ternary(BIG(k),' ','*')); end
    fprintf('%9.2f%9.2f%9s\n', min(rj), max(rj), ternary(ok,'OK','**违规**'));
end
fprintf('\n8 个状态点: ');
for k = 1:8
    fprintf('%d)x=%+.0fmm,v=%+.1f,a=%+.2f,w=%+.1f  ', k, P(1,k)*1000, P(2,k), P(3,k), P(4,k));
    if mod(k,2)==0, fprintf('\n              '); end
end
fprintf('\n');

%% ===== ③ Gamma 跨振幅（描述函数，5 个振幅）=====
fprintf('\n===== ③ Gamma @3.96Hz 跨振幅（要求 <=1.00 且不随振幅上升）=====\n');
[bo,ao] = td(0.5, DT);
As = [0.01 0.02 0.04 0.08 0.15];
f  = 3.96;  N = round(30/DT);  t = (0:N-1)'*DT;
Gm = zeros(NN, numel(As));
for i = 1:NN
    for j = 1:numel(As)
        A = As(j);  th = A*sin(2*pi*f*t);
        ua = zeros(1,3); ya = zeros(1,3); O = zeros(4,N);
        for n = 1:N
            al = -th(n) + BA;  ua = [al ua(1:2)];
            ya = [(bo(1)*ua(1)+bo(2)*ua(2)+bo(3)*ua(3)-ao(2)*ya(1)-ao(3)*ya(2)), ya(1:2)];
            O(:,n) = [0; 0; al; OM*ya(1)/10];
        end
        out = extractdata(predict(NET{i}, dlarray(single(O),'CB')));
        m_  = tanh(double(out(1,:)));  w = round(N*0.5):N;
        Gm(i,j) = 2*abs((m_(w)*cos(2*pi*f*t(w)))/numel(w) + ...
                     1i*(m_(w)*sin(2*pi*f*t(w)))/numel(w))/A;
    end
end
G1 = Gm(1,:);                                    % 克隆做基准
fprintf('%-7s','A(rad)'); fprintf('%8.2f', As); fprintf('%9s%9s%9s\n','最大','升幅','判定');
for i = 1:NN
    rel  = Gm(i,:)./G1;
    rise = rel(end) - rel(1);
    ok   = max(rel) <= 1.00 && rise <= 0.30;
    fprintf('%-7s', nets{i,1}); fprintf('%8.2f', rel);
    fprintf('%9.2f%+9.2f%9s\n', max(rel), rise, ternary(ok,'OK','**违规**'));
end

%% ===== ④ 静态平衡点（数值解 mu(x,0,BA,0)=0）=====
fprintf('\n===== ④ 静态平衡点（数值解，非线性网络不能用线性公式）=====\n');
fprintf('%-7s%12s%12s%14s\n','net','平衡点(mm)','原点mu','线性预测(mm)');
for i = 1:NN
    lo = -0.34; hi = 0.34;
    flo = mu(NET{i},[lo;0;BA;0]);
    for it = 1:60
        mid = (lo+hi)/2;  fm = mu(NET{i},[mid;0;BA;0]);
        if sign(fm) == sign(flo), lo = mid; flo = fm; else, hi = mid; end
    end
    xeq = (lo+hi)/2;
    % 原点雅可比 -> 线性预测（对比用）
    h = 1e-3;  g = zeros(1,4);
    for k = 1:4
        ep = zeros(4,1); ep(k)= h;  em = zeros(4,1); em(k)= -h;
        g(k) = (mu(NET{i},ep) - mu(NET{i},em))/(2*h);
    end
    xlin = -(abs(g(3))/abs(g(1)))*BA;
    fprintf('%-7s%12.1f%12.4f%14.1f\n', nets{i,1}, xeq*1000, ...
            mu(NET{i},zeros(4,1)), xlin*1000);
end

%% ===== ⑥ 角速度通道绝对回路增益（S10 的 bang-bang 就是这里漏掉的）=====
% 教训：③ 的 Gamma 是【相对克隆的比值】，比值把绝对量级约掉了。
% 0.5Hz 微分器在 3.96Hz 衰减到 1.58%，掩盖了 S10 高出 2.2 倍的 Kw。
% mu 中来自角速度的贡献 = Kw*(omega/10)；tanh 输入超过 2 即饱和。
% 真机实测 omega_rms：克隆 0.026 / S8 0.027（均稳定）；S10 0.542（bang-bang）
fprintf('\n===== ⑥ 角速度通道绝对回路增益（Kw<=32 且 Kw*w_rms/10<0.3）=====\n');
OMRMS = 0.030;                     % 真机稳态 omega rms（克隆/S8 实测约 0.027）
fprintf('%-7s%9s%14s%10s\n','net','Kw','Kw*w_rms/10','判定');
for i = 1:NN
    h = 1e-3;  ep = zeros(4,1); ep(4)= h;  em = zeros(4,1); em(4)= -h;
    Kw = abs((mu(NET{i},ep) - mu(NET{i},em))/(2*h));
    drv = Kw*OMRMS/10;
    ok  = Kw <= 32 && drv < 0.3;
    fprintf('%-7s%9.2f%14.3f%10s\n', nets{i,1}, Kw, drv, ternary(ok,'OK','**违规**'));
end

%% ===== ⑤ 原点结构 + 饱和 =====
fprintf('\n===== ⑤ 原点雅可比结构 =====\n');
fprintf('%-7s%9s%9s%9s%9s%9s%11s\n','net','Kx','Ka','Kw','Kw/Ka','Ka/Kx','饱和%');
for i = 1:NN
    h = 1e-3;  g = zeros(1,4);
    for k = 1:4
        ep = zeros(4,1); ep(k)= h;  em = zeros(4,1); em(k)= -h;
        g(k) = (mu(NET{i},ep) - mu(NET{i},em))/(2*h);
    end
    % 饱和占比：在工作域内随机采样
    rng(1);  M = 20000;
    Sm = [(rand(1,M)-0.5)*2*0.25; (rand(1,M)-0.5)*2*0.30
          (rand(1,M)-0.5)*2*0.25; (rand(1,M)-0.5)*2*0.30];
    o  = extractdata(predict(NET{i}, dlarray(single(Sm),'CB')));
    sat = 100*mean(abs(tanh(double(o(1,:)))) > 0.99);
    fprintf('%-7s%9.2f%9.2f%9.2f%9.2f%9.2f%11.1f\n', nets{i,1}, ...
            abs(g(1)), abs(g(3)), abs(g(4)), abs(g(4))/abs(g(3)), ...
            abs(g(3))/abs(g(1)), sat);
end
fprintf('\n完成。\n');

function [b,a] = td(f0,DT)
  z=0.9; wn=2*pi*f0; K=2/DT; a0=K^2+2*z*wn*K+wn^2;
  b=[wn^2*K 0 -wn^2*K]/a0; a=[1 2*(wn^2-K^2)/a0 (K^2-2*z*wn*K+wn^2)/a0];
end

function s = ternary(c,a,b), if c, s=a; else, s=b; end, end
