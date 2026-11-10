% S17 复核（2026-11-06）：新种子 61–76，之前没有看过。只读，结果写 logs/S17_train/verify.md。
% 和原样原厂块、S16 第 640 局同表。原厂块不加低通和积分（用户决定）。
% v1 对原厂块静置摆角和角点偏悲观，这两项不和网络比；×0.4 是过程噪声倍数，只用于看小车晃动。
% 用法：matlab -batch "PICK='logs/S17_train/snap_ep00xxx.mat'; s17_verify"
cd(fileparts(mfilename('fullpath')));
P = v1_params(); DT = P.DT; A25 = 1 - exp(-2*pi*25*DT);
FS = [2.0 2.5 3.0 3.5 3.96 4.5 5.0 6.0];
SD = 61:76; SD4 = 61:64; nS = numel(SD);

% 原厂块（K、±35 N、F/Volt Tustin、×1000/12、小信号 ×1.2、±1000，同 lqrs12s8_sim.m）
Rm = 2.12; rmp = 0.013745; Kt = 0.08505; Kb = 0.11693; Beq = 30.0; Mt = 0.4719 + 0.1679;
b1 = Rm*rmp^2*Mt; b0 = Rm*rmp^2*Beq + Kt*Kb; a1 = Kt*rmp*Mt; a0 = Kt*rmp*Beq; KT = 2/DT;
P.LQ.K = [-38.7298 164.2110 -76.3417 41.0570];
P.LQ.FV = [b1*KT + b0, b0 - b1*KT, a0 - a1*KT, a1*KT + a0];

T = {'原厂块（原样）', '',                                  1,   1;
     'S16 第 640 局',  'logs/S16_train/snap_ep00640.mat',   A25, 1;
     'S17',            PICK,                                A25, 1};
CC = cell(1,size(T,1));
for i = 1:size(T,1)
    c = struct('name',T{i,1},'a',T{i,3},'k',T{i,4},'imax',4,'Ti',0.5,'lqr',isempty(T{i,2}),'W',[],'n5',false);
    if ~c.lqr, S = load(T{i,2}); c.W = getW(S.actorNet); c.n5 = size(c.W.W1,2) == 5; end
    CC{i} = c;
end
iL = 1; iS = numel(CC);
fid = fopen(fullfile('logs','S17_train','verify.md'),'w'); w = @(varargin) [fprintf(fid,varargin{:}), fprintf(varargin{:})];
t0 = tic;
w('# S17 复核（v1，%s）\n\n脚本 `s17_verify.m`，新种子 %d–%d，选中的快照：`%s`。\n', datestr(now,'yyyy-mm-dd HH:MM'), SD(1), SD(end), PICK);

%% 1. 居中与不撞
w('\n## 1. 居中与不撞（150 s，正负零偏各 %d 局）\n\n|控制器|零偏 mrad|符号|40–60 s x 均值|最差一局|100–150 s x 均值|全程最远 mm|I 贴限幅占比|撞|\n|---|---|---|---|---|---|---|---|---|\n', nS);
for i = 1:numel(CC)
    c = CC{i};
    for b = [10 20 30]
        for sgn = [1 -1]
            xa = nan(1,nS); xb = xa; pk = xa; sat = xa; nf = 0;
            for j = 1:nS
                R = runC(c,P,150,SD(j),sgn*b/1000,[],struct());
                pk(j) = 1000*max(abs(R.X));
                if ~R.ok, nf = nf+1; continue; end
                xa(j) = 1000*mean(R.X(R.t>=40&R.t<60)); xb(j) = 1000*mean(R.X(R.t>=100)); sat(j) = mean(abs(R.I(R.t>=100)) >= 0.99*c.imax);
            end
            w('|%s|%d|%s|%+.0f|%.0f|%+.0f|%.0f|%.0f%%|%d|\n', c.name, b, char(43+(1-sgn)), mean(xa,'omitnan'), max(abs(xa)), mean(xb,'omitnan'), max(pk), 100*mean(sat,'omitnan'), nf);
        end
    end
end
fprintf('第 1 节 %.0f s\n', toc(t0));

%% 2. 静置、总晃动、|S|、脉冲
[PR0,TW0] = probe(FS,10,DT); [PRb,TWb] = probe(FS,40,DT);
PK = zeros(1,round(68/DT)); tk = [20 32 44 56]; sg = [1 -1 1 -1];
for j = 1:4, PK(round(tk(j)/DT)+(1:round(0.1/DT))) = 700*sg(j); end
w('\n## 2. 静置、晃动、|S|、脉冲（种子 %d–%d）\n\n|控制器|θ std °|PWM std|ΔPWM|x std mm|慢晃 mm|×0.4 θ °|×0.4 x std mm|×0.4 慢晃 mm|零偏 0 \\|S\\| 峰（频率）|−20 mrad \\|S\\| 峰（频率）|2.5 Hz \\|S\\||脉冲 θ 偏离 °|恢复 s|探测撞（零偏 0、−20）|脉冲撞|最远 mm|\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n', SD(1), SD(end));
FL = {};
for i = 1:numel(CC)
    c = CC{i}; st = nan(nS,5); s4 = nan(nS,3); Sa = nan(nS,numel(FS)); Sb = Sa; rc = []; pk = []; n0 = 0; n2 = 0; np = 0; mx = nan(1,nS);
    for j = 1:nS
        sd = SD(j);
        R = runC(c,P,60,sd,0,[],struct());
        if R.ok, m = R.t>=15; st(j,:) = [rad2deg(std(R.AL(m))), std(R.U(m)), sqrt(mean(diff(R.U(m)).^2)), 1000*std(R.X(m)), sway(R,15,60)]; end
        R = runC(c,P,60,sd,0,[],struct('ds',0.4));
        if R.ok, m = R.t>=15; s4(j,:) = [rad2deg(std(R.AL(m))), 1000*std(R.X(m)), sway(R,15,60)]; end
        R = runC(c,P,TW0+64,sd,0,PR0,struct());
        if R.ok, Sa(j,:) = sens(R,PR0,FS,TW0); else, n0 = n0+1; FL{end+1} = failStr(c,'零偏 0 探测',sd,R); end %#ok<SAGROW>
        R = runC(c,P,TWb+64,sd,-0.020,PRb,struct());
        if R.ok, Sb(j,:) = sens(R,PRb,FS,TWb); else, n2 = n2+1; FL{end+1} = failStr(c,'−20 mrad 探测',sd,R); end %#ok<SAGROW>
        R = runC(c,P,68,sd,0,PK,struct()); mx(j) = 1000*max(abs(R.X));
        if ~R.ok, np = np+1; FL{end+1} = failStr(c,'脉冲',sd,R); end %#ok<SAGROW>
        for q = 1:4
            [rr,dd] = recov(R,tk(q)); rc(end+1) = rr; pk(end+1) = dd; %#ok<SAGROW>
        end
    end
    m = mean(st,1,'omitnan'); m4 = mean(s4,1,'omitnan'); sa = mean(Sa,1,'omitnan'); sb = mean(Sb,1,'omitnan'); [p0,k0] = max(sa); [p2,k2] = max(sb);
    w('|%s|%.2f|%.1f|%.1f|%.1f|%.1f|%.2f|%.1f|%.1f|%.2f（%g Hz）|%.2f（%g Hz）|%.2f|%.1f|%.2f|%d、%d|%d|%.0f|\n', c.name, m(1),m(2),m(3),m(4),m(5), m4(1),m4(2),m4(3), p0,FS(k0), p2,FS(k2), sa(2), mean(pk,'omitnan'), mean(rc,'omitnan'), n0, n2, np, max(mx));
end
if ~isempty(FL), w('\n明细：\n\n'); for q = 1:numel(FL), w('- %s\n', FL{q}); end, end
fprintf('第 2 节 %.0f s\n', toc(t0));

%% 3. 11 角点按零偏拆开
w('\n## 3. 11 角点（种子 %d–%d，64 s，含 2.5/3/3.5 Hz 探测）\n\n撞或倒的局数，每格满分 %d。\n\n|控制器|零偏 0|−20 mrad|+20 mrad|\n|---|---|---|---|\n', SD4(1), SD4(end), 11*numel(SD4));
CO = {struct('tal',0.015); struct('tal',0.025); struct('ka',1.40); struct('ka',1.89); struct('fa',3.7); struct('ta',0.009); struct('kf',0.6); struct('kf',1.0); struct('Jp',0.00262); struct('tal',0.025,'ta',0.009,'fa',3.7); struct()};
[PRc,TWc] = probe([2.5 3.0 3.5],40,DT);
for i = 1:numel(CC)
    c = CC{i}; nf = zeros(1,3); BAs = [0 -0.020 0.020];
    for k = 1:numel(CO)
        Pk = P; fn = fieldnames(CO{k}); for q = 1:numel(fn), Pk.(fn{q}) = CO{k}.(fn{q}); end
        for sd = SD4
            for q = 1:3
                R = runC(c,Pk,TWc+24,sd,BAs(q),PRc,struct()); nf(q) = nf(q) + ~R.ok;
            end
        end
    end
    w('|%s|%d|%d|%d|\n', c.name, nf(1), nf(2), nf(3));
end
fprintf('第 3 节 %.0f s\n', toc(t0));

%% 4. 判据没覆盖的工况
w('\n## 4. 运行中零偏跳变（第 40 s，120 s，%d 局系综均值）\n\n|控制器|前→后 mrad|撞/局|单局最远 mm|系综均值最远 mm|系综均值回到 ±20 mm 内用时 s|跳变后 θ 最大 °|\n|---|---|---|---|---|---|---|\n', nS);
JP = [0 -20; 0 20; -20 0; 20 0]; N120 = round(120/DT);
for i = [iL iS]
    c = CC{i};
    for q = 1:size(JP,1)
        nf = 0; pk = nan(1,nS); th = nan(1,nS); XX = nan(nS,N120);
        for j = 1:nS
            R = runC(c,P,120,SD(j),JP(q,1)/1000,[],struct('tsw',40,'BA2',JP(q,2)/1000)); nf = nf + ~R.ok;
            pk(j) = 1000*max(abs(R.X)); th(j) = rad2deg(max(abs(R.TH(R.t>=40))));
            if R.ok, XX(j,:) = R.X; end
        end
        xm = movmean(1000*mean(XX,1,'omitnan'), round(4/DT)); tt = (1:N120)*DT; aft = tt >= 40;
        k = find(abs(xm) > 20 & aft, 1, 'last');
        if isempty(k), ts = 0; elseif tt(k) >= 118, ts = NaN; else, ts = tt(k+1)-40; end
        w('|%s|%+d→%+d|%d/%d|%.0f|%.0f|%.0f|%.1f|\n', c.name, JP(q,1), JP(q,2), nf, nS, max(pk), max(abs(xm(aft))), ts, max(th));
    end
end
w('\n## 5. 同次切换（种子 %d–%d：原厂块 0–60 s → S17 60–210 s → 原厂块 210–270 s；切入时 I 清零，低通从当前 PWM 接着走）\n\n|零偏 mrad|撞/局|切入后 10 s θ 最大 °|S17 段小车最远 mm|130–210 s x 均值|切回后 20 s 最远 mm|\n|---|---|---|---|---|---|\n', SD4(1), SD4(end));
for b = [0 -20 20]
    nf = 0; th = []; px = []; xe = []; pb = [];
    for sd = SD4
        R = runC(CC{iL},P,270,sd,b/1000,[],struct('seq',{{CC{iL},CC{iS},CC{iL}}},'tseq',[60 210])); nf = nf + ~R.ok;
        m = R.t>=60&R.t<70; if any(m), th(end+1) = rad2deg(max(abs(R.TH(m)))); end %#ok<SAGROW>
        m = R.t>=60&R.t<210; if any(m), px(end+1) = 1000*max(abs(R.X(m))); end %#ok<SAGROW>
        if R.ok
            xe(end+1) = 1000*mean(R.X(R.t>=130&R.t<210)); pb(end+1) = 1000*max(abs(R.X(R.t>=210&R.t<230))); %#ok<SAGROW>
        end
    end
    w('|%+d|%d/%d|%.1f|%.0f|%+.0f|%.0f|\n', b, nf, numel(SD4), max(th), max(px), mean([xe NaN],'omitnan'), max([pb NaN]));
end
w('\n## 6. 600 s 长跑（种子 %d–%d）\n\n|零偏 mrad|撞/局|100–200 s x std|500–600 s x std|100–200 s θ std °|500–600 s θ std °|500–600 s x 均值|\n|---|---|---|---|---|---|---|\n', SD4(1), SD4(end));
for b = [0 -20 20]
    v = nan(numel(SD4),5); nf = 0;
    for j = 1:numel(SD4)
        R = runC(CC{iS},P,600,SD4(j),b/1000,[],struct()); nf = nf + ~R.ok;
        if ~R.ok, continue; end
        m1 = R.t>=100&R.t<200; m2 = R.t>=500;
        v(j,:) = [1000*std(R.X(m1)) 1000*std(R.X(m2)) rad2deg(std(R.AL(m1))) rad2deg(std(R.AL(m2))) 1000*mean(R.X(m2))];
    end
    m = mean(v,1,'omitnan');
    w('|%+d|%d/%d|%.1f|%.1f|%.2f|%.2f|%+.1f|\n', b, nf, numel(SD4), m(1),m(2),m(3),m(4),m(5));
end
fprintf('第 4–6 节 %.0f s\n', toc(t0));

%% 7. predict 路径抽查（和上机同一路径：single、predict）
c = CC{iS}; S = load(PICK); net = S.actorNet;
rng(1); N = 2000; O = [(rand(1,N)-.5)*0.66; (rand(1,N)-.5)*1.0; (rand(1,N)-.5)*0.2; (rand(1,N)-.5)*1.0; (rand(1,N)-.5)*8];
a = double(extractdata(predict(net, dlarray(single(O),'CB')))); hb = actFwd(c.W, O);
Is = -4:1:4; muI = zeros(size(Is));
for q = 1:numel(Is), z = actFwd(c.W,[0;0;0;0;Is(q)]); muI(q) = tanh(z(1)); end
w('\n## 7. 网络文件\n\n- 手写前向和 predict 的 μ 最大差 %.2e（%d 个随机点，|I| ≤ 4）。\n- μ(I)（其余输入 0，I = −4…+4）：%s\n', max(abs(tanh(a(1,:)) - tanh(hb(1,:)))), N, sprintf('%+.3f ',muI));
w('\n总用时 %.0f s。\n', toc(t0));
fclose(fid);
fprintf('写入完成\n');

%% ---------------- 子函数 ----------------
function R = runC(c, P, T, seed, BA, PR, o)
    DT = P.DT; N = round(T/DT); if isempty(PR), PR = zeros(1,N); end
    ds = 1; if isfield(o,'ds'), ds = o.ds; end
    ksw = -1; if isfield(o,'tsw'), ksw = round(o.tsw/DT); end
    seq = isfield(o,'seq'); jc = 1; if seq, kq = round(o.tseq/DT); c = o.seq{1}; end
    rng(seed); th0 = (rand-0.5)*0.02;
    E = v1_env_init(P, th0, randi(numel(P.noise))-1, BA, 0);
    X = zeros(1,N); TH = X; AL = X; U = X; II = X; y = 0; I = 0; fu = 0; fy = 0; ok = true;
    for k = 1:N
        if k == ksw, E.BA = o.BA2; end
        if seq && jc <= numel(kq) && k == kq(jc)+1, jc = jc+1; c = o.seq{jc}; I = 0; end   % 切段：I 清零，y 接着走
        [ob,E,al,xq] = v1_env_observe(E,P);
        I = max(min(I + ob(1)*DT/c.Ti, c.imax), -c.imax);
        if c.lqr || seq                                                        % 切段时原厂块一直在算，和真机一样
            uF = -(P.LQ.K*[xq; al; ob(2)*5; ob(4)*10/P.OM]); uF = max(min(uF,35),-35);
            fy = (P.LQ.FV(1)*uF + P.LQ.FV(2)*fu - P.LQ.FV(3)*fy)/P.LQ.FV(4); fu = uF;
            pl = fy*1000/12; if abs(pl) <= 100, pl = 1.2*pl; end
            pl = max(min(pl,1000),-1000);
        end
        if c.lqr
            u = pl;
        else
            if c.n5, z = actFwd(c.W,[ob;I]); else, z = actFwd(c.W,ob); end
            u = 1000*max(min(c.k*tanh(z(1)),1),-1);
        end
        y = y + c.a*(u - y);
        pwm = max(min(y + PR(k),1000),-1000);
        E = v1_env_step(E,P,pwm,ds);
        X(k) = E.s(1); TH(k) = E.s(3); AL(k) = al; U(k) = pwm; II(k) = I;
        if abs(E.s(3)) > 0.5 || abs(E.s(1)) > 0.33, ok = false; break; end
    end
    R.X = X(1:k); R.TH = TH(1:k); R.AL = AL(1:k); R.U = U(1:k); R.I = II(1:k); R.t = (1:k)*DT; R.ok = ok;
end

function [PR, TW] = probe(F, TW, DT)
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

function S = sens(R, PR, F, TW)
    S = nan(1,numel(F)); t = R.t; PR = PR(1:numel(t));
    for i = 1:numel(F)
        m = t >= TW + 8*(i-1) + 1 & t < TW + 8*i;
        if sum(m) < 1000, continue; end
        e = exp(-1j*2*pi*F(i)*t(m));
        S(i) = abs(sum((R.U(m) - mean(R.U(m))).*e)) / abs(sum(PR(m).*e));
    end
end

function [rr, dd] = recov(R, t0)
    % 恢复时间：|读数 − 脉冲前均值| ≤ 0.05 rad 连续 1 s 的起点；dd：3 s 内最大偏离（度）
    rr = NaN; dd = NaN;
    pre = R.t>=t0-5 & R.t<t0; m3 = R.t>=t0 & R.t<t0+3; m = R.t>=t0 & R.t<t0+10;
    if sum(m) < 2500, return; end
    a0 = mean(R.AL(pre)); dd = rad2deg(max(abs(R.AL(m3)-a0)));
    ok = abs(R.AL(m)-a0) <= 0.05; n = 0; tm = R.t(m);
    for q = 1:numel(ok)
        if ok(q), n = n+1; else, n = 0; end
        if n >= 250, rr = tm(q-249)-t0; break; end
    end
end

function s = sway(R, t0, t1)
    m = R.t >= t0 & R.t < t1;
    x = 1000*(R.X(m) - mean(R.X(m))); n = numel(x); h = 0.5 - 0.5*cos(2*pi*(0:n-1)/(n-1));
    Xf = abs(fft(x.*h)).^2 / (sum(h.^2)*n); f = (0:n-1)/(n*0.004);
    s = sqrt(2*sum(Xf(f >= 0.1 & f <= 0.5)));
end

function s = failStr(c, what, sd, R)
    kind = '小车撞 ±330 mm'; if abs(R.TH(end)) > 0.5, kind = '摆杆倒'; end
    s = sprintf('%s，%s，种子 %d：第 %.1f s %s（x %+.0f mm，θ %+.1f°）', c.name, what, sd, R.t(end), kind, 1000*R.X(end), rad2deg(R.TH(end)));
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
