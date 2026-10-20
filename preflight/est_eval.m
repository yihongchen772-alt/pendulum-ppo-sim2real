% EST 估计器带宽试验的评估（方案与判据 logs/EST_test/plan.md）：s12_pre.m 的副本，
% 对照改为 S12（0.5 Hz），每个候选用自己的角速度微分器频率；最后按事前判据自动判定。
% 用法：matlab -batch "est_eval"；结果 logs/EST_test/eval.md
% ---- 以下为 s12_pre.m 原注释 ----
% S12 上机前体检（校准仿真 v1）：S12 候选 vs S8 deploy，同种子配对。
% 口径与真机分析一致：摆角用读数（含延迟、噪声、量化），PWM 为最终限幅后的值。
%   ① 标称静置：零偏 0（相当于外环已收敛），60 s 取 15–60 s，种子 1–4：
%      θ std、PWM std、ΔPWM RMS、x 均值/std、2.4–3.3 Hz 功率占比、1–8 Hz 主峰占比
%   ② 标称 |S|：步进正弦 150 PWM，2/2.5/3/3.5/3.96/4.5/5/6 Hz 各 8 s（去掉每段第 1 s），种子 1–4
%   ③ 参数角点（v1 不确定的方向，各 2 种子）：40 s 静置（取 10–40 s）+ 2.5/3/3.5 Hz 探测
%   ④ 脉冲：700 PWM × 0.1 s，20/32/44/56 s 交替正负，种子 1–2；θ 最大偏离（3 s 内）、
%      恢复（|α−脉冲前均值| ≤ 0.05 rad 持续 1 s）、是否倒（|θ|>25°）
%   ⑤ 零偏 −20 mrad、无外环：x 均值（40 s 取 15–40 s，种子 1–2）
% 用法：matlab -batch "s12_pre"；默认候选 = actor_net_S12_best.mat + logs/S12_train/snap_*.mat
% 结果：logs/S12_train/preflight.md
cd(fileparts(mfilename('fullpath')));
P = v1_params(); DT = P.DT;
FS = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0];

P0 = P;
C = {'S12（0.5 Hz）', 'actor_net_S12_deploy.mat', 0.5};
dd = dir(fullfile('logs','EST_test','EST_f*_s*'));
for i = 1:numel(dd)
    fn = fullfile(dd(i).folder, dd(i).name, 'actor_net_final.mat');
    if ~dd(i).isdir || contains(dd(i).name,'smoke') || ~exist(fn,'file'), continue; end
    tk = regexp(dd(i).name, 'EST_f(\d+)_s(\d+)', 'tokens', 'once');
    fw = str2double(tk{1})/100;
    C(end+1,:) = {sprintf('EST %.1f Hz s%s', fw, tk{2}), fn, fw}; %#ok<SAGROW>
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

out = fullfile('logs','EST_test','eval.md');
fid = fopen(out,'w');
w = @(varargin) fprintf(fid, varargin{:});
w('# EST 估计器带宽试验评估（v1 校准仿真，%s）\n\n', datestr(now,'yyyy-mm-dd HH:MM'));
w('候选 %d 个；同种子配对，S12（0.5 Hz）为对照；每个网络用自己的角速度微分器。摆角为读数口径。\n', nc);

res = struct([]);
for c = 1:nc
    P = P0; [P.BO, P.AO] = td2c(C{c,3}, DT);   % 每个候选用自己的估计器
    S = load(C{c,2}); W = getW(S.actorNet);
    r = struct('name', C{c,1});
    % ① 标称静置
    st = zeros(4,7);
    for sd = 1:4
        R = runSim(W, P, 60, sd, 0, []);
        st(sd,:) = statSeg(R, 15, 60);
    end
    r.st = st;
    % ② 标称 |S|
    Sm = nan(4,numel(FS));
    for sd = 1:4
        [PR,TW] = probeVec(FS, 10, DT);
        R = runSim(W, P, TW + 8*numel(FS), sd, 0, PR);
        if R.ok, Sm(sd,:) = calcS(R, PR, FS, TW); end
    end
    r.S = Sm;
    % ③ 角点
    nk = size(CO,1); r.co = nan(nk,6);   % θ std, PWM std, 2.4–3.3 占比, |S|2.5, |S|3, |S|3.5
    for k = 1:nk
        Pk = P; fn = fieldnames(CO{k,2});
        for j = 1:numel(fn), Pk.(fn{j}) = CO{k,2}.(fn{j}); end
        v = nan(2,6);
        for sd = 1:2
            [PR,TW] = probeVec([2.5 3.0 3.5], 40, DT);
            R = runSim(W, Pk, TW + 24, sd, 0, PR);
            if ~R.ok, continue; end
            s1 = statSeg(R, 10, 40);
            v(sd,:) = [s1(1) s1(2) s1(6) calcS(R, PR, [2.5 3.0 3.5], TW)];
        end
        r.co(k,:) = mean(v,1,'omitnan'); r.cofail(k) = sum(isnan(v(:,1)));
    end
    % ④ 脉冲
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
        fell = fell + (~R.ok || any(abs(R.TH) > deg2rad(25)));   % 倒的局数（共 2 局）
    end
    r.pulse = [mean(pk) max(pk) mean(rc,'omitnan') fell];
    % ⑤ 零偏 −20 mrad、无外环
    xb = nan(1,2);
    for sd = 1:2
        R = runSim(W, P, 40, sd, -0.020, []);
        if R.ok, s1 = statSeg(R, 15, 40); xb(sd) = s1(4); end
    end
    r.xbias = mean(xb,'omitnan');
    res = [res r]; %#ok<AGROW>
    fprintf('%s 完成\n', C{c,1});
end

% ---- 写表 ----
w('\n## ① 标称静置（零偏 0，15–60 s，种子 1–4 均值 ± 种子间 std）\n\n');
w('|控制器|θ std °|PWM std|ΔPWM RMS|x 均值 mm|x std mm|2.4–3.3 Hz 占比|主峰占比|\n|---|---|---|---|---|---|---|---|\n');
for c = 1:nc
    m = mean(res(c).st,1); s = std(res(c).st,0,1);
    w('|%s|%.2f ± %.2f|%.1f ± %.1f|%.1f ± %.1f|%+.1f ± %.1f|%.1f ± %.1f|%.0f%% ± %.0f|%.0f%% ± %.0f|\n', res(c).name, ...
        m(1),s(1),m(2),s(2),m(3),s(3),m(4),s(4),m(5),s(5),m(6),s(6),m(7),s(7));
end
w('\n## ② 标称 |S|（150 PWM，种子 1–4 均值）\n\n|控制器|%s|峰值|\n|---|%s---|\n', sprintf('%g Hz|',FS), repmat('---|',1,numel(FS)));
for c = 1:nc
    m = mean(res(c).S,1,'omitnan');
    w('|%s|%s%.2f|\n', res(c).name, sprintf('%.2f|',m), max(m));
end
w('\n## ③ 参数角点（2 种子均值）：θ std ° / PWM std / 2.4–3.3 Hz 占比 / |S| 2.5·3·3.5 Hz\n\n');
w('|角点|%s\n|---|%s\n', sprintf('%s|',res.name), repmat('---|',1,nc));
for k = 1:size(CO,1)
    w('|%s|', CO{k,1});
    for c = 1:nc
        v = res(c).co(k,:);
        if isnan(v(1)), w('倒|'); continue; end
        w('%.2f / %.0f / %.0f%% / %.1f·%.1f·%.1f%s|', v, repmat('（倒1）',1,res(c).cofail(k)>0));
    end
    w('\n');
end
w('\n## ④ 脉冲 700 PWM × 0.1 s（8 次）与 ⑤ 零偏 −20 mrad 无外环\n\n|控制器|θ 偏离均值 °|θ 偏离最大 °|恢复 s（≤0.05 rad 持续 1 s）|倒（局/2）|零偏 −20 mrad 时 x 均值 mm|\n|---|---|---|---|---|---|\n');
for c = 1:nc
    p = res(c).pulse;
    w('|%s|%.1f|%.1f|%.2f|%d|%+.0f|\n', res(c).name, p(1), p(2), p(3), p(4), res(c).xbias);
end
% ---- 事前判据（logs/EST_test/plan.md）自动判定，对照 = 第 1 个（S12）----
b = res(1); bS = mean(b.S,1,'omitnan'); bst = mean(b.st,1); bsd = std(b.st,0,1);
w('\n## 判定（事前判据，见 plan.md；对照 S12）\n\n|候选|① 2.5 Hz |S| 降幅 / 峰值|② θ、PWM 不变差|③ ΔPWM 增幅|④ 角点|⑤ 脉冲|结论|\n|---|---|---|---|---|---|---|\n');
for c = 2:nc
    r = res(c); cS = mean(r.S,1,'omitnan'); cst = mean(r.st,1); csd = std(r.st,0,1);
    d25 = 1 - cS(2)/bS(2); k1 = d25 >= 0.20 && max(cS) <= max(bS);
    k2 = cst(1) <= bst(1) + bsd(1) + csd(1) && cst(2) <= bst(2) + bsd(2) + csd(2);
    dD = cst(3)/bst(3) - 1; k3 = dD <= 0.15;
    k4 = all(~isnan(r.co(:,1))) && all(r.cofail == 0) && all(r.co(:,3) < 60);
    k5 = r.pulse(4) == 0 && r.pulse(3) <= 1.5*b.pulse(3);
    yn = {'否','是'};
    w('|%s|%.0f%%（%.2f 对 %.2f）/ %.2f 对 %.2f %s|θ %.2f 对 %.2f，PWM %.0f 对 %.0f %s|%+.0f%% %s|%s|恢复 %.2f 对 %.2f s，倒 %d %s|**%s**|\n', ...
        r.name, 100*d25, cS(2), bS(2), max(cS), max(bS), yn{k1+1}, cst(1), bst(1), cst(2), bst(2), yn{k2+1}, ...
        100*dD, yn{k3+1}, yn{k4+1}, r.pulse(3), b.pulse(3), r.pulse(4), yn{k5+1}, ...
        repmat('有收益（进第二轮）',1,k1&&k2&&k3&&k4&&k5) + "" );
end
w('\n结论栏为空 = 至少一条不满足 ⇒ 没有收益。\n');
fclose(fid);
fprintf('写入 %s\n', out);

%% ---------------- 子函数 ----------------
function R = runSim(W, P, T, seed, BA, PR)
    N = round(T/P.DT); if isempty(PR), PR = zeros(1,N); end
    rng(seed); th0 = (rand-0.5)*0.02;
    E = v1_env_init(P, th0, randi(numel(P.noise))-1, BA);
    U = zeros(1,N); AL = U; X = U; TH = U; ok = true;
    for k = 1:N
        [o,E,al] = v1_env_observe(E,P);
        oa = actFwd(W,o);
        pwm = max(min(tanh(oa(1))*1000 + PR(k), 1000), -1000);
        E = v1_env_step(E,P,pwm);
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

function [b, a] = td2c(f0, DT)
    z = 0.9; wn = 2*pi*f0; K = 2/DT; a0 = K^2 + 2*z*wn*K + wn^2;
    b = [wn^2*K 0 -wn^2*K]/a0; a = [1 2*(wn^2 - K^2)/a0 (K^2 - 2*z*wn*K + wn^2)/a0];
end

function W = getW(net)
    V = net.Learnables.Value;
    W.W1=undl(V{1}); W.b1=undl(V{2}); W.W2=undl(V{3}); W.b2=undl(V{4}); W.W3=undl(V{5}); W.b3=undl(V{6});
end

function v = undl(v)
    if isa(v,'dlarray'), v = extractdata(v); end
    v = double(v);
end

function oa = actFwd(W,o)
    h = max(W.W1*o + W.b1, 0); h = max(W.W2*h + W.b2, 0); oa = W.W3*h + W.b3;
end
