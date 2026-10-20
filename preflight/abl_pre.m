% 上机前体检（与 s8_pre.m 同四项）：E/C 候选 vs 克隆 / S8dep / S7(失败样本)。2026-09-18
clc; BA=-0.0101; DT=0.004; z=0.9; wn=2*pi*0.5; K=2/DT; a0=K^2+2*z*wn*K+wn^2;
bo=[wn^2*K 0 -wn^2*K]/a0; ao=[1 2*(wn^2-K^2)/a0 (K^2-2*z*wn*K+wn^2)/a0];
nets={'克隆','actor_net_lqr_clone.mat';'S8dep','actor_net_S8_deploy.mat';'E102best','actor_net_ABL_E_s102_best.mat';'E102fin','actor_net_ABL_E_s102.mat';'E101best','actor_net_ABL_E_s101_best.mat';'E103best','actor_net_ABL_E_s103_best.mat';'C102best','actor_net_ABL_C_s102_best.mat';'B102best','actor_net_ABL_B_s102_best.mat';'D102best','actor_net_ABL_D_s102_best.mat';'D103best','actor_net_ABL_D_s103_best.mat';
      'S7(失败)','actor_net_S7_deploy.mat'};
fprintf('\n===== ① 位置回复力 mu(x,0,b_alpha,0) =====\n');
fprintf('%-9s','x(mm)'); xs=[-230 -175 -100 -50 0 50 100 175 230];
for v=xs, fprintf('%7d',v); end; fprintf('   单调?\n');
for i=1:size(nets,1)
  S=load(nets{i,2}); net=S.actorNet; fprintf('%-9s',nets{i,1}); m=zeros(1,numel(xs));
  for k=1:numel(xs)
    o=extractdata(predict(net,dlarray(single([xs(k)/1000;0;BA;0]),'CB')));
    m(k)=tanh(double(o(1)))*1000; fprintf('%7.0f',m(k));
  end
  if all(diff(m)>0), md='是'; else, md='否 <=='; end
  fprintf('   %s\n', md);
end
fprintf('\n===== ② Gamma 随振幅（S7 就栽在这）=====\n');
AMPS=[0.010 0.020 0.040 0.060 0.090]; F=3.95; T=4.0;
Ng=round(T/DT); tg=(0:Ng-1)'*DT; h0=round(Ng*0.5); Lw=Ng-h0+1;
O=zeros(4,numel(AMPS)*Lw);
for ia=1:numel(AMPS)
  th=AMPS(ia)*sin(2*pi*F*tg); uu=zeros(1,3); yy=zeros(1,3); Og=zeros(4,Ng);
  for n=1:Ng
    al=-th(n); uu=[al uu(1:2)];
    yy=[(bo(1)*uu(1)+bo(2)*uu(2)+bo(3)*uu(3)-ao(2)*yy(1)-ao(3)*yy(2)), yy(1:2)];
    Og(:,n)=[0;0;al;2.0*yy(1)/10];
  end
  O(:,(ia-1)*Lw+1:ia*Lw)=Og(:,h0:end);
end
CC=cos(2*pi*F*tg(h0:end))'; SS=sin(2*pi*F*tg(h0:end))';
G=zeros(4,numel(AMPS));
fprintf('%-9s','A(rad)'); for a=AMPS, fprintf('%9.3f',a); end; fprintf('%9s%9s\n','最大','升');
for i=1:size(nets,1)
  S=load(nets{i,2}); o=tanh(double(extractdata(predict(S.actorNet,dlarray(single(O),'CB'))))); o=o(1,:);
  for ia=1:numel(AMPS)
    s=o((ia-1)*Lw+1:ia*Lw); G(i,ia)=2*sqrt(mean(s.*CC)^2+mean(s.*SS)^2)/AMPS(ia);
  end
  r=G(i,:)./G(1,:); fprintf('%-9s',nets{i,1});
  for j=1:numel(AMPS), fprintf('%9.2f',r(j)); end
  fprintf('%9.2f%+9.2f\n',max(r),r(end)-r(1));
end
fprintf('\n===== ③ 带符号推力（8点）=====\n');
PTS=[-0.23 -0.10 0.10 0.23 0 0 0 0; zeros(1,8); 0 0 0 0 0.10 -0.10 0.04 -0.04; zeros(1,8)];
AREF=[-0.4673 -0.2004 +0.2109 +0.4725 -0.8502 +0.8707 -0.3389 +0.3431];
fprintf('%-9s %7s %7s %7s %7s %7s %7s %7s %7s %8s\n','网络','x-.23','x-.10','x+.10','x+.23','a+.10','a-.10','a+.04','a-.04','MIN');
for i=1:size(nets,1)
  S=load(nets{i,2}); o=extractdata(predict(S.actorNet,dlarray(single(PTS),'CB')));
  r=tanh(double(o(1,:)))./AREF; fprintf('%-9s',nets{i,1});
  fprintf('%8.2f',r); fprintf('%9.2f\n',min(r));
end
fprintf('\n===== ④ 静态平衡点 + 饱和 =====\n');
xs2=linspace(-0.34,0.34,681);
[X,V,AL,W]=ndgrid(linspace(-0.2,0.2,9),linspace(-0.5,0.5,5),linspace(-0.15,0.15,13),linspace(-2,2,7));
Q=[X(:)';V(:)'/5;AL(:)';W(:)'/10];
for i=1:size(nets,1)
  S=load(nets{i,2}); net=S.actorNet;
  QQ=[xs2; zeros(1,numel(xs2)); BA*ones(1,numel(xs2)); zeros(1,numel(xs2))];
  o=extractdata(predict(net,dlarray(single(QQ),'CB'))); mu=tanh(double(o(1,:)));
  cr=find(sign(mu(1:end-1)).*sign(mu(2:end))<0);
  xr=NaN; if ~isempty(cr), c=cr(1); xr=(xs2(c)-mu(c)*(xs2(c+1)-xs2(c))/(mu(c+1)-mu(c)))*1000; end
  o2=extractdata(predict(net,dlarray(single(Q),'CB'))); m2=tanh(double(o2(1,:)));
  fprintf('  %-9s 平衡点 %+7.1fmm   零点%d个   饱和占比 %.1f%%\n',nets{i,1},xr,numel(cr),100*mean(abs(m2)>0.99));
end
