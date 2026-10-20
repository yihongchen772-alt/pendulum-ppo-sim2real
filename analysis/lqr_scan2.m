clc; Mc=0.4719; Mp=0.1679; Lp=0.425/2; Beq=30; Bp=0.024; g=9.81; Jp=0.01;
M=[Mp+Mc,-Mp*Lp;-Mp*Lp,Mp*Lp^2+Jp];
cal=M\[0;Mp*Lp*g]; cxd=M\[-Beq;0]; cad=M\[0;Bp]; cF=M\[1;0];
A=[0 0 1 0;0 0 0 1;0 cal(1) cxd(1) cad(1);0 cal(2) cxd(2) cad(2)]; B=[0;0;cF(1);cF(2)];
K0=[-38.7298 164.2110 -76.3417 41.0570]; n0=norm(K0); BA=-0.0101;
fprintf('\n降低 qa（角度权重），R=2、qx=3000 固定\n\n');
fprintf('%9s%10s%10s%10s%12s   %s\n','qa','Ka/Kx','增益','Kad/Ka','预测偏移','K');
for qa=[5000 3000 2000 1200 800 500 300]
  K=lqr(A,B,diag([3000 qa 1 1]),2); r=abs(K(2)/K(1)); gn=norm(K)/n0;
  fprintf('%9d%10.2f%10.2f%10.3f%10.1fmm   [%7.2f %7.2f %7.2f %7.2f]\n',qa,r,gn,K(4)/K(2),-r*BA*1000,K);
end
fprintf('\n加大阻尼权重 qxd/qad（R=2, qx=3000, qa=5000 固定）\n\n');
fprintf('%9s%9s%10s%10s%10s   %s\n','qxd','qad','Ka/Kx','增益','Kad/Ka','K');
for w=[1 20 100 400 1000]
  K=lqr(A,B,diag([3000 5000 w w]),2); r=abs(K(2)/K(1));
  fprintf('%9d%9d%10.2f%10.2f%10.3f   [%7.2f %7.2f %7.2f %7.2f]\n',w,w,r,norm(K)/n0,K(4)/K(2),K);
end
fprintf('\n═══ 组合搜索：Ka/Kx<3.2 且 增益<1.00 ═══\n');
fprintf('%7s%7s%7s%7s%9s%8s%10s   %s\n','qx','qa','qxd','qad','Ka/Kx','增益','预测偏移','K');
c=0;
for qx=[3000 8000 15000 30000]
 for qa=[5000 2000 1000 500]
  for w=[1 50 300]
    K=lqr(A,B,diag([qx qa w w]),2); r=abs(K(2)/K(1)); gn=norm(K)/n0;
    if r<3.2 && gn<1.00
      c=c+1; fprintf('%7d%7d%7d%7d%9.2f%8.2f%9.1fmm   [%7.2f %7.2f %7.2f %7.2f]\n',qx,qa,w,w,r,gn,-r*BA*1000,K);
    end
  end
 end
end
if c==0, fprintf('  无\n'); end
