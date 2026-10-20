function ol_center_sim(stage, base_file)
%**************************************************************************
% 慢速外环居中：仿真验证。直接调用要上机的 fcn_S8_OL.m（逐步闭环），
% 所以验证的就是部署代码本身，包括接口换算（V_SIGN、OMEGA_SCALE、度/弧度）。
%
% 参数扫描时，把 fcn_S8_OL.m 里标了 "% @名字" 的常数行替换后另存一份临时副本
% 再调用 —— 被测逻辑与部署逻辑逐字相同，只差被扫的那个常数。
%
% 仿真器与训练/kx_causal_test.m 相同：量化、两个二阶微分器、角速度 x2、
% 角度零偏 BA、Jp=0.0025、F=pwm/42、持续过程扰动 2.0。另可加库仑摩擦 FC（N）。
%
%   ol_center_sim(1)  符号 + 增益扫描（A1+B1 两段）
%   ol_center_sim(2)  零偏扫描、脉冲定标、完整时间表（含摩擦）
%**************************************************************************
if nargin < 1, stage = 1; end
if nargin < 2, base_file = 'fcn_S8_OL.m'; end   % 上机版文件（stage 5 原样跑）
here = fileparts(mfilename('fullpath')); cd(here);
TMP = fullfile(tempdir, 'olvar'); if ~exist(TMP, 'dir'), mkdir(TMP); end
addpath(TMP);
BASE = fileread(fullfile(here, base_file));

P.DT = 0.004; P.Q_TH = (360/2048)*pi/180; P.Q_X = 0.658/5268;
pr.Jp = 0.0025; pr.OM = 2.0; pr.P2F = 42; pr.BA = -0.0101; pr.FC = 0;
[pr.bo, pr.ao] = td2(0.5, P.DT); [pr.bv, pr.av] = td2(1.0, P.DT);
DIST = 2.0;
makeFwd(TMP);   % 快速前向 + 与 predict 的逐点核对

switch stage
case 0   % 计时
    tic; R = runSim(makeVariant(BASE, struct(), 'tim', TMP), pr, P, DIST, 10, 1); el = toc;
    fprintf('10 s 仿真用时 %.1f s（%d 步）\n', el, R.n);

case 1
    fprintf('=== 1. 符号：A1 纯 S8 -> B1 开外环（零偏 BA=%.4f）===\n', pr.BA);
    hdr();
    for KI = [0.03 -0.03]
        R = runSim(makeVariant(BASE, struct('K_I', KI), 'sgn', TMP), pr, P, DIST, 240, 11);
        row(sprintf('K_I=%+.3f', KI), R, P, 120);
    end
    fprintf('\n=== 2. 增益扫描（BA=%.4f，同一随机种子）===\n', pr.BA);
    hdr();
    for KI = [0.01 0.02 0.04 0.08 0.16]
        R = runSim(makeVariant(BASE, struct('K_I', KI), sprintf('k%d', round(KI*1000)), TMP), pr, P, DIST, 240, 11);
        row(sprintf('K_I=%.3f', KI), R, P, 120);
    end

case 3   % 过程曲线：B1 段每 5 s 一行（低通 x mm / aref mrad）
    KIs = [0.01 0.02 0.04 0.08];
    for j = 1:numel(KIs)
        RR{j} = runSim(makeVariant(BASE, struct('K_I', KIs(j)), 'tr', TMP), pr, P, DIST, 240, 11);
    end
    fprintf('%6s', 't(s)'); fprintf('   K_I=%.2f: x mm / aref mrad', KIs); fprintf('\n');
    for tt = 120:5:215
        fprintf('%6d', tt - 120);
        for j = 1:numel(KIs)
            R = RR{j}; xl = lpf(R.x, 1 - exp(-2*pi*0.2*P.DT)); [~, i] = min(abs(R.t - tt));
            fprintf('   %13.1f / %7.2f         ', xl(i)*1000, R.aref(i)*1000);
        end
        fprintf('\n');
    end
    for j = 1:numel(KIs)
        [tc, os] = convStats(RR{j}, P, 2, 120); fprintf('K_I=%.2f：aref 到终值 90%% 用 %.1f s，小车反向过冲 %.1f mm\n', KIs(j), tc, os);
    end

case 4   % 脉冲定标：幅度 x 宽度
    fprintf('%-8s%8s%10s%12s%10s\n', '幅度', '宽度s', 'θ偏离°', '小车位移mm', '恢复 s');
    for AMP = [300 500 700 1000]
        for DUR = [0.05 0.10 0.20]
            R = runSim(makeVariant(BASE, struct('K_I', 0.015, 'KICK_AMP', AMP, 'KICK_DUR', DUR), 'kk', TMP), pr, P, DIST, 115, 31);
            K = kickStats(R, P, 1, 120, 100);
            fprintf('%-8d%8.2f%10.2f%12.1f%10.2f\n', AMP, DUR, K.thpk, K.dx, K.rec);
        end
    end

case 5   % 上机版文件原样（不覆盖任何参数）跑完整时间表：ol_center_sim(5, 'fcn_S8_OLc.m')
    names = {'A1', 'B1', 'A2', 'B2', 'P'};
    for BA = [-0.0101 -0.0192]
        q = pr; q.BA = BA;
        R = runSim(makeVariant(BASE, struct(), 'dep', TMP), q, P, DIST, 575, 41);
        fprintf('\n=== %s 原样，零偏 %.4f rad ===\n', base_file, BA);
        if ~isnan(R.fail), fprintf('!! 第 %.1f s 失败（倒或撞限位）\n', R.fail); end
        fprintf('%-6s%10s%9s%9s%10s%10s\n', '段', 'x均值mm', 'θstd°', 'θ峰°', 'PWMstd', 'aref末');
        for s = 1:4
            M = segStats(R, P, 120, s, 50, 95);
            fprintf('%-6s%10.1f%9.2f%9.2f%10.1f%10.4f\n', names{s}, M.xmean, M.thstd, M.thpk, M.pwmstd, M.aref);
        end
        M = segStats(R, P, 120, 5, 0, 20);
        fprintf('P 段探测前（480–500 s）：小车 %.1f mm，aref %.4f rad\n', M.xmean, M.aref);
        fprintf('脉冲：');
        for s = 1:4, K = kickStats(R, P, s, 120, 100); fprintf('%s θ峰 %.2f° 恢复 %.2f s；', names{s}, K.thpk, K.rec); end
        S = probeStats(R, P);
        fprintf('\nP 段：小车最大偏离 %.0f mm；仿真 |S|：', S.xmax); fprintf('%.2fHz %.2f  ', [S.f; S.mag]); fprintf('\n');
    end
case 2
    KI = str2double(getenv('OL_KI')); if isnan(KI), KI = 0.04; end
    fprintf('=== 3. 零偏扫描（K_I=%.3f）：真机零偏未知，看外环能否都拉回来 ===\n', KI);
    hdr();
    for BA = [-0.020 -0.010 0 0.010 0.020]
        q = pr; q.BA = BA;
        R = runSim(makeVariant(BASE, struct('K_I', KI), 'ba', TMP), q, P, DIST, 240, 21);
        row(sprintf('BA=%+.3f', BA), R, P, 120);
    end
    fprintf('\n=== 4. 脉冲定标：A1 段第 100 s 的 +脉冲（40 ms）===\n');
    fprintf('%-10s%10s%12s%12s\n', '幅度PWM', 'θ峰值°', '小车位移mm', '恢复 s');
    for AMP = [150 250 350 450]
        R = runSim(makeVariant(BASE, struct('K_I', KI, 'KICK_AMP', AMP), 'kick', TMP), pr, P, DIST, 115, 31);
        K = kickStats(R, P, 1, 120, 100);
        fprintf('%-10d%10.2f%12.1f%12.2f\n', AMP, K.thpk, K.dx, K.rec);
    end
    AMP = str2double(getenv('OL_KICK')); if isnan(AMP), AMP = 250; end
    for FC = [0 1.0]
        q = pr; q.FC = FC;
        fprintf('\n=== 5. 完整时间表（K_I=%.3f，脉冲 %d PWM，库仑摩擦 %.1f N）===\n', KI, AMP, FC);
        R = runSim(makeVariant(BASE, struct('K_I', KI, 'KICK_AMP', AMP), 'full', TMP), q, P, DIST, 575, 41);
        if ~isnan(R.fail), fprintf('!! 第 %.1f s 失败（倒或撞限位）\n', R.fail); end
        fprintf('%-6s%10s%9s%9s%9s%10s%10s%10s%10s\n', '段', 'x均值mm', 'x std', 'θstd°', 'θ峰°', 'PWMstd', 'ΔPWM', 'aref末', '慢漂std');
        names = {'A1', 'B1', 'A2', 'B2'};
        for s = 1:4
            M = segStats(R, P, 120, s, 50, 95);
            fprintf('%-6s%10.1f%9.1f%9.2f%9.2f%10.1f%10.2f%10.4f%10.2f\n', names{s}, M.xmean, M.xstd, M.thstd, M.thpk, M.pwmstd, M.dpwm, M.aref, M.slow);
        end
        fprintf('脉冲：');
        for s = 1:4
            K = kickStats(R, P, s, 120, 100);
            fprintf('%s θ峰 %.2f° 恢复 %.2f s；', names{s}, K.thpk, K.rec);
        end
        fprintf('\n');
        [tc, os] = convStats(R, P, 2, 120, 0.0050); fprintf('B1 收敛 %.1f s，反向过冲 %.1f mm；', tc, os);
        [tc, os] = convStats(R, P, 4, 120, 0.0050); fprintf('B2 收敛 %.1f s，反向过冲 %.1f mm\n', tc, os);
        a1 = segStats(R, P, 120, 1, 50, 95);
        tb = backStats(R, P, 3, 120, a1.xmean/1000);
        fprintf('A2 外环清零后回到 A1 平衡点（±5 mm）用时 %.1f s\n', tb);
        S = probeStats(R, P);
        fprintf('P 段：小车最大偏离 %.0f mm；仿真 |S|：', S.xmax);
        fprintf('%.2fHz %.2f  ', [S.f; S.mag]); fprintf('\n');
        save(fullfile(here, sprintf('ol_sim_full_FC%d.mat', round(FC*10))), 'R', 'KI', 'AMP', 'FC');
    end
end
end

%% ======================= 仿真 =======================
function name = makeVariant(base, ov, tag, TMP)
    txt = base; f = fieldnames(ov);
    for i = 1:numel(f)
        pat = ['^(\s*\w+\s*=\s*)[-+\d\.eE]+(;\s*%\s*@' f{i} '\>)'];
        assert(numel(regexp(txt, pat, 'lineanchors')) == 1, '参数行 @%s 没找到或不唯一', f{i});
        txt = regexprep(txt, pat, ['$1' sprintf('%.6g', ov.(f{i})) '$2'], 'lineanchors');
    end
    persistent cnt; if isempty(cnt), cnt = 0; end; cnt = cnt + 1;
    name = sprintf('fcnv_%s_%d', tag, cnt);
    net2 = sprintf(['dlState   = dlarray(single(state), ''CB'');\n' ...
                    'out_actor = extractdata(predict(actorNet, dlState));']);
    assert(numel(strfind(txt, net2)) == 1, '网络调用行没找到');
    txt = strrep(txt, net2, 'out_actor = s8fwd(single(state));   % 仿真专用：与 predict 逐点核对过');
    assert(numel(regexp(txt, '=\s*fcn_S8_OL\w*\(')) == 1, '函数定义行没找到');
    txt = regexprep(txt, '(=\s*)fcn_S8_OL\w*\(', ['$1' name '('], 'once');
    fid = fopen(fullfile(TMP, [name '.m']), 'w'); fwrite(fid, txt, 'char'); fclose(fid);
    clear(name); rehash;
end

function R = runSim(fname, pr, P, DIST, T_END, seed)
    rng(seed); fh = str2func(fname); clear(fname);
    N = round(T_END / P.DT);
    x = 0; xd = 0; th = (rand - 0.5) * 0.01; thd = 0; st = initFilt();
    L = zeros(6, N); fail = NaN; n = N;
    for k = 1:N
        [hw, st, al] = makeObsHW(th, x, st, pr, P);
        [pwm, ~, ~, ~, ~, aref] = fh(hw(1), hw(2), hw(3), hw(4));
        d = dyn([x; xd; th; thd], pwm, pr);
        d(4) = d(4) + DIST * randn();
        x = x + P.DT*d(1); xd = xd + P.DT*d(2); th = th + P.DT*d(3); thd = thd + P.DT*d(4);
        L(:, k) = [k*P.DT; x; th; al; pwm; aref];
        if abs(th) > 0.5 || abs(x) > 0.33, fail = k*P.DT; n = k; break; end
    end
    L = L(:, 1:n);
    R.t = L(1,:); R.x = L(2,:); R.th = L(3,:); R.al = L(4,:); R.pwm = L(5,:); R.aref = L(6,:);
    R.n = n; R.fail = fail;
end

function [hw, st, al] = makeObsHW(theta, x, st, pr, P)
    % 仿真观测 -> 真机 fcn 的四个输入（与部署接口逐项对应）
    th_q = round(theta / P.Q_TH) * P.Q_TH; x_q = round(x / P.Q_X) * P.Q_X;
    al = -th_q + pr.BA;
    st.ua = [al st.ua(1:2)];
    st.ya = [(pr.bo(1)*st.ua(1) + pr.bo(2)*st.ua(2) + pr.bo(3)*st.ua(3) ...
              - pr.ao(2)*st.ya(1) - pr.ao(3)*st.ya(2)), st.ya(1:2)];
    st.ux = [x_q st.ux(1:2)];
    st.yx = [(pr.bv(1)*st.ux(1) + pr.bv(2)*st.ux(2) + pr.bv(3)*st.ux(3) ...
              - pr.av(2)*st.yx(1) - pr.av(3)*st.yx(2)), st.yx(1:2)];
    v = st.yx(1); om = pr.OM * st.ya(1);
    hw = [x_q, al*180/pi, -v, om/(360/2048)];   % cartpos(m), rodAngle(deg), cartVel(V_SIGN=-1), rodVel(OMEGA_SCALE)
end

function st = initFilt()
    st.ua = zeros(1,3); st.ya = zeros(1,3); st.ux = zeros(1,3); st.yx = zeros(1,3);
end

function ds = dyn(s, pwm, pr)
    Mc = 0.4719; Mp = 0.1679; l = 0.2125; g = 9.81; dmp = 0.024;
    xd = s(2); th = s(3); thd = s(4);
    F = pwm / pr.P2F - pr.FC * tanh(xd / 0.01); ml = Mp*l; ml2 = Mp*l^2;
    M = [Mc + Mp, ml*cos(th); ml*cos(th), pr.Jp + ml2];
    Fv = [F - 30*xd + ml*thd^2*sin(th); -dmp*thd + Mp*g*l*sin(th)];
    a = M \ Fv; ds = [xd; a(1); thd; a(2)];
end

function [b, a] = td2(f0, DT)
    z = 0.9; wn = 2*pi*f0; K = 2/DT; a0 = K^2 + 2*z*wn*K + wn^2;
    b = [wn^2*K 0 -wn^2*K]/a0; a = [1 2*(wn^2 - K^2)/a0 (K^2 - 2*z*wn*K + wn^2)/a0];
end

function makeFwd(TMP)
    S = load('actor_net_S8_deploy.mat'); net = S.actorNet;
    L = net.Layers; types = arrayfun(@(l) class(l), L, 'UniformOutput', false);
    fc = find(contains(types, 'FullyConnected'));
    assert(numel(L) == 6 && numel(fc) == 3 && all(contains(types([3 5]), 'ReLU')), '网络结构不是 4-64-64-2 ReLU');
    assert(strcmp(L(1).Normalization, 'none'), '输入层带归一化');
    W = cell(1,3); B = cell(1,3);
    for i = 1:3
        nm = L(fc(i)).Name; T = net.Learnables;
        W{i} = single(extractdata(T.Value{strcmp(T.Layer, nm) & strcmp(T.Parameter, 'Weights')}));
        B{i} = single(extractdata(T.Value{strcmp(T.Layer, nm) & strcmp(T.Parameter, 'Bias')}));
    end
    W1 = W{1}; b1 = B{1}; W2 = W{2}; b2 = B{2}; W3 = W{3}; b3 = B{3};
    save(fullfile(TMP, 's8w.mat'), 'W1', 'b1', 'W2', 'b2', 'W3', 'b3');
    fid = fopen(fullfile(TMP, 's8fwd.m'), 'w');
    fprintf(fid, ['function out = s8fwd(s)\npersistent W1 b1 W2 b2 W3 b3\nif isempty(W1)\n' ...
        '    S = load(fullfile(fileparts(mfilename(''fullpath'')), ''s8w.mat''));\n' ...
        '    W1 = S.W1; b1 = S.b1; W2 = S.W2; b2 = S.b2; W3 = S.W3; b3 = S.b3;\nend\n' ...
        'h = max(W1*s + b1, 0); h = max(W2*h + b2, 0); out = W3*h + b3;\nend\n']);
    fclose(fid); clear s8fwd; rehash;
    rng(7); X = [(rand(1,2000)-0.5)*0.4; (rand(1,2000)-0.5)*0.4; (rand(1,2000)-0.5)*0.2; (rand(1,2000)-0.5)*0.6];
    a = extractdata(predict(net, dlarray(single(X), 'CB'))); b = zeros(size(a), 'single');
    for i = 1:2000, b(:, i) = s8fwd(single(X(:, i))); end
    d = max(abs(tanh(double(a(1,:))) - tanh(double(b(1,:)))));
    fprintf('快速前向 vs predict：2000 个随机状态，mu 最大差 %.2e\n', d);
    assert(d < 1e-5, '快速前向与 predict 不一致');
end

%% ======================= 指标 =======================
function hdr()
    fprintf('%-12s%11s%11s%10s%10s%10s%10s%10s%10s\n', '配置', 'A1 x mm', 'B1 x mm', 'B1收敛s', '过冲mm', 'θstd A/B', '', 'PWM A/B', 'aref末');
end

function row(tag, R, P, TS)
    if ~isnan(R.fail), fprintf('%-12s 第 %.1f s 失败\n', tag, R.fail); return; end
    a = segStats(R, P, TS, 1, 50, 95); b = segStats(R, P, TS, 2, 50, 95);
    [tc, os] = convStats(R, P, 2, TS, 0.0050);
    fprintf('%-12s%11.1f%11.1f%10.1f%10.1f%10.2f%10.2f%10.1f/%-6.1f%10.4f\n', tag, a.xmean, b.xmean, tc, os, a.thstd, b.thstd, a.pwmstd, b.pwmstd, b.aref);
end

function M = segStats(R, P, TS, s, w0, w1)
    t0 = (s - 1) * TS; m = R.t >= t0 + w0 & R.t < t0 + w1;
    M.xmean = mean(R.x(m))*1000; M.xstd = std(R.x(m))*1000;
    M.thstd = std(R.th(m))*180/pi; M.thpk = max(abs(R.th(m)))*180/pi;
    M.pwmstd = std(R.pwm(m)); M.dpwm = sqrt(mean(diff(R.pwm(m)).^2));
    M.aref = R.aref(find(m, 1, 'last'));
    xl = lpf(R.x, 1 - exp(-2*pi*0.5*P.DT)); M.slow = std(xl(m))*1000;   % 0.5 Hz 以下的慢漂
end

function [tc, os] = convStats(R, P, s, TS, ~)
    % 收敛：aref 到达终值（段内 50–95 s 均值）90% 的时间；过冲：只看脉冲之前（段内 0–95 s）
    % 低通 x 越过 0 之后的最大反向偏移
    t0 = (s - 1) * TS; m = find(R.t >= t0 & R.t < t0 + 95);
    fin = mean(R.aref(R.t >= t0 + 50 & R.t < t0 + 95));
    i = find(abs(R.aref(m) - fin) < 0.1 * abs(fin), 1); tc = NaN; if ~isempty(i), tc = R.t(m(i)) - t0; end
    xl = lpf(R.x, 1 - exp(-2*pi*0.2*P.DT)); x0 = xl(m(1));
    os = max(0, max(-sign(x0) * xl(m))) * 1000;
end

function tb = backStats(R, P, s, TS, xeq)
    t0 = (s - 1) * TS; xl = lpf(R.x, 1 - exp(-2*pi*0.2*P.DT));
    m = find(R.t >= t0 & R.t < t0 + TS); i = find(abs(xl(m) - xeq) < 0.005, 1);
    if isempty(i), tb = NaN; else, tb = R.t(m(i)) - t0; end
end

function K = kickStats(R, P, s, TS, at)
    % 摆角峰值 = 脉冲后 3 s 内真实摆角相对扰动前 5 s 均值的最大偏离；恢复 = 回到扰动前噪声 3 倍带内并保持 1 s
    t0 = (s - 1) * TS + at; pre = R.t >= t0 - 5 & R.t < t0; m3 = R.t >= t0 & R.t < t0 + 3; m = R.t >= t0 & R.t < t0 + 10;
    K.thpk = max(abs(R.th(m3) - mean(R.th(pre))))*180/pi;
    K.dx = max(abs(R.x(m3) - mean(R.x(pre))))*1000;
    band = max(3*std(R.al(pre)), 0.01);
    idx = find(m); ok = abs(R.al(idx) - mean(R.al(pre))) <= band; need = round(1/P.DT); K.rec = NaN; c = 0;
    for i = 1:numel(idx)
        if ok(i), c = c + 1; else, c = 0; end
        if c >= need, K.rec = R.t(idx(i - need + 1)) - t0; break; end
    end
end

function S = probeStats(R, P)
    t0 = 4*120 + 20; F = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0]; S.f = F; S.mag = zeros(size(F));
    mm = R.t >= 4*120 & R.t < 4*120 + 90; S.xmax = max(abs(R.x(mm)))*1000;
    for i = 1:8
        m = R.t >= t0 + (i-1)*8 + 1.0 & R.t < t0 + i*8;   % 去掉爬升段
        tk = R.t(m) - (t0 + (i-1)*8); p = 150*sin(2*pi*F(i)*tk); u = R.pwm(m);   % 150 与 fcn 的 P_AMP 一致
        e = exp(-1j*2*pi*F(i)*tk); S.mag(i) = abs(sum(u.*e)) / abs(sum(p.*e));
    end
end

function y = lpf(x, a)
    y = filter(a, [1 -(1 - a)], x, (1 - a)*x(1));
end
