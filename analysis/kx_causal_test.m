%**************************************************************************
% 廉价因果验证：Kx 抬高 => 小车参与度上升 => 模态频率下降 => 抖动变大？
%
% 【为什么做】真机数据给出一条相关链，但只有 n=6 且有反例（S11 的 Kx 比 S10
% 更高却好得多）。这里【不训练】，直接拿 S8 的网络人为改 Kx，其余全部不动，
% 看仿真里小车参与度与模态频率是否随之变化 —— 单变量，能定因果方向。
%
% 【做法】在 tanh 之前加一个线性 x 项：  mu' = tanh( z(obs) + beta*x )
% 则原点处  d(mu')/dx = (1-mu^2)*( dz/dx + beta )，扫 beta 即可命中目标 Kx。
% 其它三个通道完全不动。
%
% 【测什么】与真机同口径：theta std、x std、主峰频率、峰功率占比、
%          小车参与度 x/theta @峰频（mm 每 rad）。
%
% 【重要】若仿真复现不出这条趋势，那也是结论 —— 说明该模态是仿真器里
% 没有的机制（与"仿真复现不出 Ms=5.59"同源），不能靠仿真去设计 S12 的约束。
%**************************************************************************
clc; rng(20260911);

%% ---------------- 真机标定参数（与训练脚本一致）----------------
P.DT=0.004; P.MAX_ACTION=1000; P.Q_TH=(360/2048)*pi/180; P.Q_X=0.658/5268;
P.THETA_LIMIT=0.2; P.X_LIMIT=0.2;
pr.Jp=0.0025; pr.OM=2.0; pr.P2F=42; pr.BA=-0.0101;
[pr.bo,pr.ao]=td2(0.5,P.DT);     % 角速度滤波器
[pr.bv,pr.av]=td2(1.0,P.DT);     % 车速滤波器
DIST_STD=2.0;                    % 持续过程扰动，与训练一致
NEP=24; NSTEP=7500;              % 24 局 x 30 秒

S=load('actor_net_S8_deploy.mat'); net=S.actorNet;

%% ---------------- 先量 S8 原本的 Kx ----------------
kx0 = measKx(net,0);
fprintf('S8 原始 Kx = %.3f\n', kx0);
fprintf('目标对照：S8 0.38 / 克隆 2.09 / S10 3.03 / S11 4.14\n\n');

%% ---------------- 扫 beta，命中若干目标 Kx ----------------
targets = [kx0 1.0 2.0 3.0 4.0 6.0];
betas = zeros(size(targets));
for i=1:numel(targets)
    betas(i) = solveBeta(net, targets(i));
end

fprintf('%-8s%8s%7s%9s%11s%10s%9s%9s%9s\n', ...
    'Kx目标','实得Kx','存活%','θstd°','居中 mm','xstd mm','主峰Hz','峰占比','x/θ');
fprintf('%s\n', repmat('-',1,80));
for i=1:numel(targets)
    R = rollout(net, betas(i), P, pr, DIST_STD, NEP, NSTEP);
    fprintf('%-8.2f%8.3f%7.0f%9.2f%11.1f%10.1f%8.2fHz%8.0f%%%9.0f\n', ...
        targets(i), measKx(net,betas(i)), R.alive*100, R.thstd, R.xmean, R.xstd, ...
        R.fpk, R.frac*100, R.ratio);
end

fprintf('\n真机实测对照（供比较）:\n');
fprintf('  S8   Kx 0.38  θ0.98°  居中 -36.4mm  主峰3.97Hz  峰占比 9.7%%  x/θ 150\n');
fprintf('  S10  Kx 3.03  θ8.44°  居中 -95.7mm  主峰2.26Hz  峰占比89.5%%  x/θ 312\n');
fprintf('  S11  Kx 4.14  θ1.68°  居中 -38.6mm  主峰2.93Hz  峰占比51.2%%  x/θ 267\n');

%% ================= 子函数 =================
function kx = measKx(net, beta)
    h=1e-4; o=zeros(4,1);
    ep=o; ep(1)=h; em=o; em(1)=-h;
    kx = (muOf(net,ep,beta)-muOf(net,em,beta))/(2*h);   % 【带符号】
    % 不取 abs：beta 很负时 |Kx| 也很大，abs 会破坏单调性使二分失效
end

function mu = muOf(net, obs, beta)
    z = extractdata(predict(net, dlarray(single(obs),'CB')));
    mu = tanh(double(z(1,:)) + beta*double(obs(1,:)));
end

function beta = solveBeta(net, target)
    lo=-5; hi=30;                    % 单调：beta 增大 -> Kx 增大
    % 量级：dmu/dx ≈ (1-mu^2)*(dz/dx + beta)，原点 mu≈0，S8 的 dz/dx≈0.38
    % 所以要 Kx=6 只需 beta≈5.6，范围取 [-5,30] 足够
    for k=1:80
        mid=(lo+hi)/2;
        if measKx(net,mid) < target, lo=mid; else, hi=mid; end
    end
    beta=(lo+hi)/2;
end

function R = rollout(net, beta, P, pr, DIST, NEP, NSTEP)
    THS=[]; XS=[]; alive=0;
    for e=1:NEP
        x=(rand-0.5)*0.06; xd=0; th=(rand-0.5)*0.04; thd=0;
        st=initFilt(); thL=zeros(1,NSTEP); xL=zeros(1,NSTEP); n=0;
        for k=1:NSTEP
            [obs,st]=makeObs(th,x,st,pr,P);
            mu = muOf(net,obs,beta);
            pwm = max(min(mu*P.MAX_ACTION,P.MAX_ACTION),-P.MAX_ACTION);
            d=dyn([x;xd;th;thd],pwm,pr);
            d(4)=d(4)+DIST*randn();
            x=x+P.DT*d(1); xd=xd+P.DT*d(2); th=th+P.DT*d(3); thd=thd+P.DT*d(4);
            if abs(th)>P.THETA_LIMIT || abs(x)>P.X_LIMIT, break; end
            n=n+1; thL(n)=th; xL(n)=x;
        end
        if n>=NSTEP*0.95
            alive=alive+1;
            m=round(n*0.3):n;                 % 去掉起始暂态
            THS=[THS thL(m)]; XS=[XS xL(m)];
        end
    end
    R.alive=alive/NEP;
    if isempty(THS), R.thstd=NaN;R.xstd=NaN;R.xmean=NaN;R.fpk=NaN;R.frac=NaN;R.ratio=NaN; return; end
    R.thstd=std(THS)*180/pi; R.xstd=std(XS)*1000; R.xmean=mean(XS)*1000;
    % 频谱（Welch 风格：分段平均）
    [f,Pt]=psdSeg(THS,P.DT); [~,Px]=psdSeg(XS,P.DT);
    b=(f>=1.0)&(f<=8.0); [~,j]=max(Pt(b)); fb=f(b); R.fpk=fb(j);
    pk=(f>R.fpk-0.3)&(f<R.fpk+0.3); tot=(f>0.2);
    R.frac=sum(Pt(pk))/sum(Pt(tot));
    i0=find(abs(f-R.fpk)==min(abs(f-R.fpk)),1);
    R.ratio=sqrt(Px(i0)/Pt(i0))*1000;
end

function [f,Pxx]=psdSeg(y,DT)
    L=4096; if numel(y)<L, L=2^floor(log2(numel(y))); end
    nseg=floor(numel(y)/L); Pxx=zeros(L/2+1,1); w=0.5*(1-cos(2*pi*(0:L-1)'/(L-1)));   % Hann 窗（不依赖工具箱）
    for i=1:nseg
        seg=y((i-1)*L+1:i*L)'; seg=seg-mean(seg);
        Y=fft(seg.*w); Pxx=Pxx+abs(Y(1:L/2+1)).^2;
    end
    Pxx=Pxx/nseg; f=(0:L/2)'/(L*DT);
end

function [obs,st]=makeObs(theta,x,st,pr,P)
    th_q=round(theta/P.Q_TH)*P.Q_TH; x_q=round(x/P.Q_X)*P.Q_X;
    al=-th_q+pr.BA;
    st.ua=[al st.ua(1:2)];
    st.ya=[(pr.bo(1)*st.ua(1)+pr.bo(2)*st.ua(2)+pr.bo(3)*st.ua(3) ...
           -pr.ao(2)*st.ya(1)-pr.ao(3)*st.ya(2)), st.ya(1:2)];
    st.ux=[x_q st.ux(1:2)];
    st.yx=[(pr.bv(1)*st.ux(1)+pr.bv(2)*st.ux(2)+pr.bv(3)*st.ux(3) ...
           -pr.av(2)*st.yx(1)-pr.av(3)*st.yx(2)), st.yx(1:2)];
    obs=[x_q; st.yx(1)/5.0; al; pr.OM*st.ya(1)/10.0];
end

function st=initFilt()
    st.ua=zeros(1,3); st.ya=zeros(1,3); st.ux=zeros(1,3); st.yx=zeros(1,3);
end

function ds=dyn(s,pwm,pr)
    Mc=0.4719; Mp=0.1679; l=0.2125; g=9.81; dmp=0.024;
    xd=s(2); th=s(3); thd=s(4);
    F=pwm/pr.P2F; ml=Mp*l; ml2=Mp*l^2;
    M=[Mc+Mp, ml*cos(th); ml*cos(th), pr.Jp+ml2];
    Fv=[F-30*xd+ml*thd^2*sin(th); -dmp*thd+Mp*g*l*sin(th)];
    a=M\Fv; ds=[xd;a(1);thd;a(2)];
end

function [b,a]=td2(f0,DT)
  z=0.9; wn=2*pi*f0; K=2/DT; a0=K^2+2*z*wn*K+wn^2;
  b=[wn^2*K 0 -wn^2*K]/a0; a=[1 2*(wn^2-K^2)/a0 (K^2-2*z*wn*K+wn^2)/a0];
end
