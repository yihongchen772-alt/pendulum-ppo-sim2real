% S15 上机前体检：s14b_pre.m 换候选和输出路径，六条判据不变，加第 7、8 条（logs/S15_train/plan.md）。
% “S14b ×1.1”是不重训的兜底；S14b 第 640 局是起点，一起算。
% 原说明：S14 上机前体检（校准仿真 v1）：logs/S14_train/plan.md 的六条判据，同种子配对。
%   零偏 0：① 静置 ② |S| ③ 11 角点 ④ 700 PWM 脉冲（五条同 logs/S13_train/plan.md 第 2 步，对照 S12 deploy）；小车 0.1–0.5 Hz 慢晃对照 S13b
%   ⑤ 居中：零偏 ±20 mrad，种子 1–4，60 s，取 40–60 s；⑥ 零偏 ±30 mrad，8 局不撞不倒
%   ⑦ 零偏 −20 mrad 收敛 40 s 后的 |S|；⑧ 11 角点在零偏 −20 mrad 下
% “S13b+手加 g=0.2”只作参考：等同外环，不算 RL 学到的，也不算候选。
% 用法：matlab -batch "s15_pre"；结果 logs/S15_train/preflight.md
cd(fileparts(mfilename('fullpath')));
P = v1_params(); DT = P.DT; P.T_I = 2.0; P.I_MAX = 1.0; A25 = 1 - exp(-2*pi*25*DT);
FS = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0];
C = {'S12 deploy',       'actor_net_S12_deploy.mat',          1,   0, 1, 2,   1;
     'S13b ep08800',     'logs/S13b_train/snap_ep08800.mat',  A25, 0, 1, 2,   1;
     'S14b ep00640',     'logs/S14b_train/snap_ep00640.mat',  A25, 0, 4, 0.5, 1;
     'S14b ×1.1',        'logs/S14b_train/snap_ep00640.mat',  A25, 0, 4, 0.5, 1.1;
     'S15 best',         'actor_net_S15_best.mat',            A25, 0, 4, 0.5, 1};
sn = dir(fullfile('logs','S15_train','snap_*.mat'));
for i = 1:numel(sn)
    C(end+1,:) = {strrep(erase(sn(i).name,'.mat'),'snap_','S15 '), fullfile(sn(i).folder, sn(i).name), A25, 0, 4, 0.5, 1}; %#ok<SAGROW>
end
nc = size(C,1);
CO = {'标称',            struct();
      '角度延迟 15 ms',   struct('tal',0.015);
      '角度延迟 25 ms',   struct('tal',0.025);
      '推力 ×1.40/1.647', struct('ka',1.40);
      '推力 ×1.89/1.647', struct('ka',1.89);
      '执行器 3.7 Hz',    struct('fa',3.7);
      '纯延迟 9 ms',      struct('ta',0.009);
      '摩擦 kf 0.6',      struct('kf',0.6);
      '摩擦 kf 1.0',      struct('kf',1.0);
      'Jp 0.00262',       struct('Jp',0.00262);
      '滞后最坏（25 ms + 9 ms + 3.7 Hz）', struct('tal',0.025,'ta',0.009,'fa',3.7)};
res = struct([]);
for c = 1:nc
    S = load(C{c,2}); W = getW(S.actorNet); W.a = C{c,3}; W.g = C{c,4}; W.imax = C{c,5}; W.Ti = C{c,6}; W.k = C{c,7};
    r = struct('name', C{c,1});
    % 零偏 0：静置与慢晃
    st = zeros(4,7); sw = zeros(1,4);
    for sd = 1:4
        R = runSim(W, P, 60, sd, 0, []); st(sd,:) = statSeg(R, 15, 60); sw(sd) = swayBand(R, 15, 60);
    end
    r.st = st; r.sway = mean(sw);
    % |S|（零偏 0、零偏 -20 mrad 收敛后）
    Sm = nan(4,numel(FS)); Sb = Sm;
    [PR0,TW0] = probeVec(FS, 10, DT); [PRb,TWb] = probeVec(FS, 40, DT);
    for sd = 1:4
        R = runSim(W, P, TW0+64, sd, 0, PR0); if R.ok, Sm(sd,:) = calcS(R, PR0, FS, TW0); end
        R = runSim(W, P, TWb+64, sd, -0.020, PRb); if R.ok, Sb(sd,:) = calcS(R, PRb, FS, TWb); end
    end
    r.S = Sm; r.Sb = Sb;
    % 11 角点（零偏 0、零偏 -20 mrad）
    nk = size(CO,1); r.co = nan(nk,6); r.cofail = zeros(1,nk); r.cobfail = zeros(1,nk);
    [PRc,TWc] = probeVec([2.5 3.0 3.5], 40, DT);
    for k = 1:nk
        Pk = P; fn = fieldnames(CO{k,2}); for j = 1:numel(fn), Pk.(fn{j}) = CO{k,2}.(fn{j}); end
        v = nan(2,6);
        for sd = 1:2
            R = runSim(W, Pk, TWc+24, sd, 0, PRc);
            if R.ok, s1 = statSeg(R, 10, 40); v(sd,:) = [s1(1) s1(2) s1(6) calcS(R, PRc, [2.5 3.0 3.5], TWc)]; end
            R = runSim(W, Pk, TWc+24, sd, -0.020, PRc); r.cobfail(k) = r.cobfail(k) + ~R.ok;
        end
        r.co(k,:) = mean(v,1,'omitnan'); r.cofail(k) = sum(isnan(v(:,1)));
    end
    % 700 PWM 脉冲
    pk = []; rc = []; fell = 0;
    for sd = 1:2
        PR = zeros(1, round(68/DT)); tk = [20 32 44 56]; sg = [1 -1 1 -1];
        for j = 1:4, PR(round(tk(j)/DT)+(1:round(0.1/DT))) = 700*sg(j); end
        R = runSim(W, P, 68, sd, 0, PR);
        for j = 1:4
            t0 = tk(j); pre = R.t>=t0-5 & R.t<t0; m3 = R.t>=t0 & R.t<t0+3; m = R.t>=t0 & R.t<t0+10;
            if sum(m) < round(10/DT), continue; end
            a0 = mean(R.AL(pre)); pk(end+1) = rad2deg(max(abs(R.AL(m3)-a0))); %#ok<SAGROW>
            ok = abs(R.AL(m)-a0) <= 0.05; need = round(1/DT); cnt = 0; rr = NaN; tm = R.t(m);
            for i = 1:numel(ok)
                if ok(i), cnt = cnt+1; else, cnt = 0; end
                if cnt >= need, rr = tm(i-need+1) - t0; break; end
            end
            rc(end+1) = rr; %#ok<SAGROW>
        end
        fell = fell + (~R.ok || any(abs(R.TH) > deg2rad(25)));
    end
    r.pulse = [mean(pk) max([pk NaN]) mean(rc,'omitnan') fell];
    % 第 8 条：脉冲种子 5–28
    r.p24 = 0;
    for sd = 5:28
        R = runSim(W, P, 68, sd, 0, PR); r.p24 = r.p24 + (~R.ok || any(abs(R.TH) > deg2rad(25)));
    end
    % 第 7 条：过程噪声 ×0.4 的总晃动，种子 13–20
    xs = nan(1,8); sw4 = xs;
    for sd = 13:20
        R = runSim(W, P, 60, sd, 0, [], 0.4); s4 = statSeg(R, 15, 60); xs(sd-12) = s4(5); sw4(sd-12) = swayBand(R, 15, 60);
    end
    r.xs4 = mean(xs); r.sw4 = mean(sw4);
    % 居中 ±20 mrad；不撞 ±30 mrad
    xm = nan(2,4); xs = nan(2,4); sgn = [1 -1]; hit30 = 0; hit20 = 0;
    for q = 1:2
        for sd = 1:4
            R = runSim(W, P, 60, sd, sgn(q)*0.020, []); hit20 = hit20 + ~R.ok;
            if R.ok, m = R.t>=40; xm(q,sd) = 1000*mean(R.X(m)); xs(q,sd) = 1000*std(R.X(m)); end
            R = runSim(W, P, 60, sd, sgn(q)*0.030, []); hit30 = hit30 + ~R.ok;
        end
    end
    r.ctr = [mean(xm(1,:),'omitnan') mean(xm(2,:),'omitnan') mean(xs(:),'omitnan') hit20]; r.hit30 = hit30;
    % 局部量
    r.kv = kvOf(W); r.gi = giOf(W, P); r.kg = kgOf(W);
    res = [res r]; %#ok<AGROW>
    fprintf('%s 完成\n', C{c,1});
end

fid = fopen(fullfile('logs','S15_train','preflight.md'),'w'); w = @(varargin) fprintf(fid, varargin{:});
w('# S15 上机前体检（v1，%s）\n\n候选 %d 个，同种子配对。S12 deploy 是五条的对照，S13b ep08800 是慢晃的对照。\n', datestr(now,'yyyy-mm-dd HH:MM'), nc);
w('\n## 零偏 0 静置（15–60 s，种子 1–4）与局部量\n\n|控制器|θ std °|PWM std|ΔPWM|x 均值 mm|x std mm|慢晃 0.1–0.5 Hz mm|2.4–3.3 Hz 占比|g 均值(最小~最大)|Kv(-200)·Kv(+200)|锚点 Kv/Kα/Kω|×0.4 x std mm|×0.4 慢晃 mm|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n');
for c = 1:nc
    m = mean(res(c).st,1); w('|%s|%.2f|%.1f|%.1f|%+.1f|%.1f|%.1f|%.0f%%|%.3f(%.3f~%.3f)|%.1f·%.1f|%.2f/%.2f/%.2f|%.1f|%.1f|\n', res(c).name, m(1),m(2),m(3),m(4),m(5),res(c).sway,m(6),res(c).gi,res(c).kv,res(c).kg,res(c).xs4,res(c).sw4);
end
w('\n## |S|（150 PWM，种子 1–4 均值）\n\n|控制器|工况|%s峰值|\n|---|---|%s---|\n', sprintf('%g Hz|',FS), repmat('---|',1,numel(FS)));
for c = 1:nc
    a = mean(res(c).S,1,'omitnan'); b = mean(res(c).Sb,1,'omitnan');
    w('|%s|零偏 0|%s%.2f|\n|%s|−20 mrad|%s%.2f|\n', res(c).name, sprintf('%.2f|',a), max(a), res(c).name, sprintf('%.2f|',b), max(b));
end
w('\n## 居中、不撞、脉冲\n\n|控制器|+20 mrad x 均值|−20 mrad x 均值|x std|±20 撞/8|±30 撞/8|脉冲 θ 偏离 °|恢复 s|脉冲倒/2|\n|---|---|---|---|---|---|---|---|---|\n');
for c = 1:nc
    q = res(c).ctr; p = res(c).pulse; w('|%s|%+.0f|%+.0f|%.0f|%d|%d|%.1f|%.2f|%d|\n', res(c).name, q(1), q(2), q(3), q(4), res(c).hit30, p(1), p(3), p(4));
end
b = mean(res(1).st,1); brec = res(1).pulse(3); bsw = res(2).sway;
w('\n## 判定（八条，S12 θ %.2f°、ΔPWM %.1f、恢复 %.2f s；S13b 慢晃 %.1f mm）\n\n', b(1), b(3), brec, bsw);
w('|控制器|1 居中 ≤50 mm|2 ±30 不撞|3a ΔPWM|3b |S| 峰|3c θ|3d 角点|3e 脉冲|4a −20 |S| 峰|4b −20 角点|5 慢晃 ≤1.25|6 Kv ≥8|7 总晃动 ≤19|8 脉冲 24 局|全过|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n');
yn = {'否','过'}; pass = false(1,nc);
for c = 3:nc
    r = res(c); m = mean(r.st,1); pk = max(mean(r.S,1)); pkb = max(mean(r.Sb,1)); nf = sum(r.cofail); sh = max(r.co(:,3));
    cx = max(abs(r.ctr(1:2))); swr = r.sway/bsw;
    t = [~isnan(cx) && cx <= 50 && r.ctr(4) == 0, r.hit30 == 0, m(3)/b(3)-1 <= -0.25, ~any(isnan(r.S(:))) && pk <= 5.0, ~any(isnan(r.st(:))) && m(1) <= 1.10*b(1), ...
         nf == 0 && ~any(isnan(r.co(:,3))) && sh < 60, r.pulse(4) == 0 && r.pulse(3) <= 1.5*brec, ~any(isnan(r.Sb(:))) && pkb <= 5.0, sum(r.cobfail) == 0, swr <= 1.25, min(r.kv) >= 8, ~isnan(r.xs4) && r.xs4 <= 19, r.p24 == 0];
    pass(c) = all(t);
    w('|%s|%s %.0f|%s %d|%s %+.0f%%|%s %.2f|%s %.2f|%s %d·%.0f%%|%s %d·%.2f|%s %.2f|%s %d|%s %.2f|%s %.1f|%s %.1f|%s %d|%s|\n', r.name, yn{t(1)+1}, cx, yn{t(2)+1}, r.hit30, yn{t(3)+1}, 100*(m(3)/b(3)-1), ...
        yn{t(4)+1}, pk, yn{t(5)+1}, m(1), yn{t(6)+1}, nf, sh, yn{t(7)+1}, r.pulse(4), r.pulse(3)/brec, yn{t(8)+1}, pkb, yn{t(9)+1}, sum(r.cobfail), yn{t(10)+1}, swr, yn{t(11)+1}, min(r.kv), yn{t(12)+1}, r.xs4, yn{t(13)+1}, r.p24, yn{all(t)+1});
end
w('\n“S14b ×1.1”是不重训的兜底。挑选规则见 logs/S15_train/plan.md，训练得分不参与。\n');
fclose(fid);
fprintf('写入完成\n');

%% ---------------- 子函数 ----------------
function R = runSim(W, P, T, seed, BA, PR, ds)
    if nargin < 7, ds = 1; end
    N = round(T/P.DT); if isempty(PR), PR = zeros(1,N); end
    rng(seed); th0 = (rand-0.5)*0.02;
    E = v1_env_init(P, th0, randi(numel(P.noise))-1, BA);
    U = zeros(1,N); AL = U; X = U; TH = U; ok = true; y = 0; I = 0;
    for k = 1:N
        [o,E,al] = v1_env_observe(E,P);
        I = max(min(I + o(1)*P.DT/W.Ti, W.imax), -W.imax);
        if W.n5, oa = actFwd(W,[o;I]); else, oa = actFwd(W,o); end
        y = y + W.a*(W.k*tanh(oa(1) + W.g*W.Ti*I)*1000 - y);    % 探测和脉冲加在低通之后
        pwm = max(min(y + PR(k), 1000), -1000);
        E = v1_env_step(E,P,pwm,ds);
        U(k) = pwm; AL(k) = al; X(k) = E.s(1); TH(k) = E.s(3);
        if abs(E.s(3)) > 0.5 || abs(E.s(1)) > 0.33, ok = false; break; end
    end
    R.U = U(1:k); R.AL = AL(1:k); R.X = X(1:k); R.TH = TH(1:k); R.t = (1:k)*P.DT; R.ok = ok;
end

function v = statSeg(R, t0, t1)
    % [θ std °, PWM std, ΔPWM RMS, x 均值 mm, x std mm, 2.4–3.3 Hz 占比 %, 1–8 Hz 主峰占比 %]
    m = R.t >= t0 & R.t < t1;
    if ~R.ok || sum(m) < 1000, v = nan(1,7); return; end
    a = R.AL(m); u = R.U(m); x = R.X(m);
    a0 = a - mean(a); n = numel(a0); hw = 0.5 - 0.5*cos(2*pi*(0:n-1)/(n-1));
    X = abs(fft(a0.*hw)).^2; X = X(1:floor(n/2)+1);
    f = (0:floor(n/2))/(n*0.004);
    tot = sum(X(f > 0.2));
    band = f > 1 & f < 8; Xb = X; Xb(~band) = 0; [~,k] = max(Xb);
    v = [rad2deg(std(a)), std(u), sqrt(mean(diff(u).^2)), 1000*mean(x), 1000*std(x), ...
         100*sum(X(f >= 2.4 & f <= 3.3))/tot, 100*sum(X(max(k-2,1):k+2))/tot];
end

function [PR, TW] = probeVec(F, TW, DT)
    N = round((TW + 8*numel(F))/DT); PR = zeros(1,N);
    for k = 1:N
        tt = k*DT;
        if tt > TW
            sg = floor((tt-TW)/8);
            if sg < numel(F)
                ts = tt - TW - sg*8; PR(k) = 150*min(ts/0.5,1)*sin(2*pi*F(sg+1)*ts);
            end
        end
    end
end

function S = calcS(R, PR, F, TW)
    S = nan(1,numel(F)); t = R.t; PR = PR(1:numel(t));
    for i = 1:numel(F)
        m = t >= TW + 8*(i-1) + 1 & t < TW + 8*i;
        if sum(m) < 1000, continue; end
        e = exp(-1j*2*pi*F(i)*t(m));
        S(i) = abs(sum((R.U(m) - mean(R.U(m))).*e)) / abs(sum(PR(m).*e));
    end
end


function s = swayBand(R, t0, t1)
    % 小车 0.1–0.5 Hz 频带 RMS（mm）：去均值、Hann 窗，按窗能量归一
    m = R.t >= t0 & R.t < t1;
    if ~R.ok || sum(m) < 1000, s = NaN; return; end
    x = 1000*(R.X(m) - mean(R.X(m))); n = numel(x); h = 0.5 - 0.5*cos(2*pi*(0:n-1)/(n-1));
    Xf = abs(fft(x.*h)).^2 / (sum(h.^2)*n); f = (0:n-1)/(n*0.004);
    s = sqrt(2*sum(Xf(f >= 0.1 & f <= 0.5)));
end

function kv = kvOf(W)
    e = 0.01; kv = zeros(1,2); xs = [-0.2 0.2];
    for q = 1:2
        a = actFwd(W, pad([xs(q);e;0;0],W)); b = actFwd(W, pad([xs(q);-e;0;0],W));
        kv(q) = (tanh(a(1)) - tanh(b(1)))/(2*e);
    end
end

function g = giOf(W, P)
    % g = ∂μ/∂I / T_I，在 x、α 随机、|I| ≤ 4 的 32 个点上算（手加增益和 4 输入网络按定义给）
    if ~W.n5, g = [W.g W.g W.g]; return; end
    rs = rng(7); K = 32; v = zeros(1,K); h = 1e-3;
    for i = 1:K
        o = [(rand-0.5)*0.3; 0; (rand-0.5)*0.06; 0; (rand-0.5)*8];
        a = actFwd(W, o + [0;0;0;0;h]); b = actFwd(W, o - [0;0;0;0;h]);
        v(i) = (tanh(a(1)) - tanh(b(1)))/(2*h)/W.Ti;
    end
    rng(rs); g = [mean(v) min(v) max(v)];
end

function k = kgOf(W)
    % 锚点平均的 [Kv Kα Kω]，同 train_rl_S15.m 的 anchorGains；4 输入网络 I 列去掉
    rs = rng(20251106); K = 64;
    A = [(rand(1,K)-0.5)*0.30; zeros(1,K); (rand(1,K)-0.5)*0.06; zeros(1,K); (rand(1,K)-0.5)*8];
    rng(rs); if ~W.n5, A = A(1:4,:); end
    h = 1e-3; c = [2 3 4]; s = [1/5 -1 -0.2]; k = zeros(1,3);
    for q = 1:3
        ep = A; ep(c(q),:) = ep(c(q),:) + h; em = A; em(c(q),:) = em(c(q),:) - h;
        a = fwMu(W,ep); b = fwMu(W,em); k(q) = s(q)*W.k*mean((a-b)/(2*h));
    end
end

function m = fwMu(W,o)
    h = max(W.W1*o + W.b1, 0); h = max(W.W2*h + W.b2, 0); y = W.W3*h + W.b3; m = tanh(y(1,:));
end

function o = pad(o,W)
    if W.n5, o = [o;0]; end
end

function W = getW(net)
    V = net.Learnables.Value;
    W.W1=undl(V{1}); W.b1=undl(V{2}); W.W2=undl(V{3}); W.b2=undl(V{4}); W.W3=undl(V{5}); W.b3=undl(V{6});
    W.n5 = size(W.W1,2) == 5;
end

function v = undl(v)
    if isa(v,'dlarray'), v = extractdata(v); end
    v = double(v);
end

function oa = actFwd(W,o)
    h = max(W.W1*o + W.b1, 0); h = max(W.W2*h + W.b2, 0); oa = W.W3*h + W.b3;
end
