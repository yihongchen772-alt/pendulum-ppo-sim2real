function calib_probe_sim()
%**************************************************************************
% 在校准后的时域仿真里做"和真机一模一样"的步进正弦探测，按同样方法算 |S|：
% 非线性网络在真实振幅下的等效增益（描述函数）与 ±1000 限幅都自然包含在内。
%   1) 复现已测过的：克隆x1（实测 200 PWM，峰处被削，5.5）、克隆x0.75（实测 4.2，未削）、v25b（≥5.3，被削）
%   2) 预测明天的：S8（150 PWM）
%   3) 9/21 的五个网络（150 PWM）：C、E 是否比 S8/B/D 更接近临界
% 被控对象四种：原模型 / 校准 4 参 / 校准 6 参（plant_fit.py，只拟合 G）/ 联合 3 参（plant_fit2.py，G + 干净 |S|，最可信）
% 环境变量 ONLY_JOINT=1 时只跑原模型与联合 3 参。
%**************************************************************************
cd(fileparts(mfilename('fullpath'))); maxNumCompThreads(1);
P.DT = 0.004; P.Q_TH = (360/2048)*pi/180; P.Q_X = 0.658/5268; DIST = 2.0;
[P.bo, P.ao] = td2(0.5, P.DT); [P.bv, P.av] = td2(1.0, P.DT);
FS = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0];
models = struct('name', {'原模型', '校准4参', '校准6参', '联合3参'}, 'ka', {1, 1.45623, 3.0, 1.60114}, 'fa', {Inf, 3.94330, 2.82191, 3.56836}, ...
                'tal', {0, 0.01827, 0.01993, 0.02030}, 'B', {30, 30, 57.3467, 30}, 'Jp', {0.0025, 0.0025, 0.00169, 0.0025});
if nargin_models_only(), models = models([1 4]); end
runs = {'克隆x1 @200',    'actor_net_lqr_clone.mat',       1.00, 200, '实测 0.76 2.85 5.54 4.52 3.23 2.06 1.66 1.25（3 Hz 47% 贴限幅）';
        '克隆x0.75 @200', 'actor_net_lqr_clone.mat',       0.75, 200, '实测  —   3.83 4.23 2.96 2.28 1.70 1.59 1.26（未饱和）';
        'v25b @200',      'actor_net_v25_best.mat',        1.00, 200, '实测 1.54 4.13 5.34 3.56 2.02 1.56 1.51 1.24（3 Hz 41% 贴限幅）';
        'S8 @150',        'actor_net_S8_deploy.mat',       1.00, 150, '明天实测';
        'ABL-B @150',     'actor_net_ABL_B_s102_best.mat', 1.00, 150, '真机 1.18°，无集中峰';
        'ABL-D @150',     'actor_net_ABL_D_s102_best.mat', 1.00, 150, '真机 1.43°，无集中峰';
        'ABL-C @150',     'actor_net_ABL_C_s102_best.mat', 1.00, 150, '真机 2.63°，3 Hz 起振';
        'ABL-E @150',     'actor_net_ABL_E_s102_best.mat', 1.00, 150, '真机 3.55°，3 Hz 起振'};
for mi = 1:numel(models)
    md = models(mi);
    fprintf('\n=== %s ===\n%-15s', md.name, '');
    fprintf('%7.2f', FS); fprintf('%8s%9s   真机\n', 'Ms', '贴限幅');
    for ri = 1:size(runs, 1)
        Wn = loadW(runs{ri, 2});
        R = probeSim(Wn, runs{ri, 3}, runs{ri, 4}, md, P, DIST, FS, 1);
        if ~R.ok, fprintf('%-15s 仿真中倒下（第 %.1f s）\n', runs{ri, 1}, R.tfail); continue; end
        fprintf('%-15s', runs{ri, 1}); fprintf('%7.2f', R.S); fprintf('%8.2f%8.0f%%   %s\n', max(R.S), 100*max(R.sat), runs{ri, 5});
    end
end
end

function tf = nargin_models_only()
    tf = ~isempty(getenv('ONLY_JOINT'));
end

function R = probeSim(W, scale, AMP, md, P, DIST, FS, seed)
    rng(seed); T = 20 + 8*numel(FS); N = round(T / P.DT);
    x = 0; xd = 0; th = (rand - 0.5) * 0.02; thd = 0; BA = -0.0101;
    ua = zeros(1,3); ya = zeros(1,3); ux = zeros(1,3); yx = zeros(1,3);
    Fa = 0; tauA = 1 / (2*pi*md.fa); dly = md.tal / P.DT; buf = th * ones(1, 16);
    U = zeros(1, N); PR = zeros(1, N); R.ok = true; R.tfail = NaN;
    for k = 1:N
        tt = k * P.DT; buf = [th buf(1:end-1)];
        i0 = floor(dly); fr = dly - i0; thm = (1 - fr) * buf(1 + i0) + fr * buf(2 + i0);
        th_q = round(thm / P.Q_TH) * P.Q_TH; x_q = round(x / P.Q_X) * P.Q_X; al = -th_q + BA;
        ua = [al ua(1:2)]; ya = [(P.bo(1)*ua(1)+P.bo(2)*ua(2)+P.bo(3)*ua(3)-P.ao(2)*ya(1)-P.ao(3)*ya(2)) ya(1:2)];
        ux = [x_q ux(1:2)]; yx = [(P.bv(1)*ux(1)+P.bv(2)*ux(2)+P.bv(3)*ux(3)-P.av(2)*yx(1)-P.av(3)*yx(2)) yx(1:2)];
        o = [x_q; yx(1)/5; al; 2.0*ya(1)/10];
        h = max(W.w{1}*o + W.b{1}, 0); h = max(W.w{2}*h + W.b{2}, 0); z = W.w{3}*h + W.b{3};
        p = 0;
        if tt > 20
            sg = floor((tt - 20) / 8) + 1;
            if sg <= numel(FS), ts = tt - 20 - (sg - 1)*8; p = AMP * min(ts / 0.5, 1) * sin(2*pi*FS(sg)*ts); end
        end
        pwm = max(min(scale * tanh(z(1)) * 1000 + p, 1000), -1000);
        Fc = md.ka * pwm / 42;
        if isinf(md.fa), Fa = Fc; else, Fa = Fa + P.DT / tauA * (Fc - Fa); end
        d = dyn([x; xd; th; thd], Fa, md.B, md.Jp); d(4) = d(4) + DIST * randn();
        x = x + P.DT*d(1); xd = xd + P.DT*d(2); th = th + P.DT*d(3); thd = thd + P.DT*d(4);
        U(k) = pwm; PR(k) = p;
        if abs(th) > 0.5 || abs(x) > 0.33, R.ok = false; R.tfail = tt; return; end
    end
    t = (1:N) * P.DT; R.S = zeros(1, numel(FS)); R.sat = zeros(1, numel(FS));
    for i = 1:numel(FS)
        m = t >= 20 + 8*(i-1) + 1 & t < 20 + 8*i; e = exp(-1j*2*pi*FS(i)*t(m));
        R.S(i) = abs(sum((U(m) - mean(U(m))) .* e)) / abs(sum(PR(m) .* e));
        R.sat(i) = mean(abs(U(m)) >= 999);
    end
end

function W = loadW(fn)
    S = load(fn); net = S.actorNet; L = net.Layers; T = net.Learnables;
    fc = find(arrayfun(@(l) isa(l, 'nnet.cnn.layer.FullyConnectedLayer'), L));
    for i = 1:3
        nm = L(fc(i)).Name;
        W.w{i} = double(extractdata(T.Value{strcmp(T.Layer, nm) & strcmp(T.Parameter, 'Weights')}));
        W.b{i} = double(extractdata(T.Value{strcmp(T.Layer, nm) & strcmp(T.Parameter, 'Bias')}));
    end
end

function ds = dyn(s, F, B, Jp)
    Mc = 0.4719; Mp = 0.1679; l = 0.2125; g = 9.81; dmp = 0.024;
    xd = s(2); th = s(3); thd = s(4); ml = Mp*l;
    M = [Mc + Mp, ml*cos(th); ml*cos(th), Jp + ml*l];
    Fv = [F - B*xd + ml*thd^2*sin(th); -dmp*thd + Mp*g*l*sin(th)];
    a = M \ Fv; ds = [xd; a(1); thd; a(2)];
end

function [b, a] = td2(f0, DT)
    z = 0.9; wn = 2*pi*f0; K = 2/DT; a0 = K^2 + 2*z*wn*K + wn^2;
    b = [wn^2*K 0 -wn^2*K]/a0; a = [1 2*(wn^2 - K^2)/a0 (K^2 - 2*z*wn*K + wn^2)/a0];
end
