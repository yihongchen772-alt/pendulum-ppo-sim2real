% 从选中的快照生成上机网络文件（2026-11-07）：S13b 第 8800 局、S17 第 320 局。
% actor_net_S17.mat、actor_net_S17_best.mat 都不是选中的网络，部署文件只从快照来。
% 同时核对：存盘后的网络，predict（和上机同一路径，single）与手写前向一致。
% 用法：matlab -batch "s17_deploy"
cd(fileparts(mfilename('fullpath')));
T = {'logs/S13b_train/snap_ep08800.mat', 'actor_net_S13b_deploy.mat', 4;
     'logs/S17_train/snap_ep00320.mat',  'actor_net_S17_deploy.mat',  5};
rng(1); N = 2000;
for i = 1:size(T,1)
    S = load(T{i,1}); actorNet = S.actorNet; save(T{i,2}, 'actorNet');
    Q = load(T{i,2}); net = Q.actorNet; W = getW(net); assert(size(W.W1,2) == T{i,3});
    o = [(rand(1,N)-0.5)*0.6; (rand(1,N)-0.5)*0.4; (rand(1,N)-0.5)*0.6; (rand(1,N)-0.5)*1.0];
    if T{i,3} == 5, o = [o; (rand(1,N)-0.5)*8]; end %#ok<AGROW>
    z = zeros(1,N); for k = 1:N, r = extractdata(predict(net, dlarray(single(o(:,k)),'CB'))); z(k) = tanh(double(r(1))); end
    h = max(W.W1*o + W.b1, 0); h = max(W.W2*h + W.b2, 0); y = W.W3*h + W.b3; zh = tanh(y(1,:));
    fprintf('%s：%s，%d 输入，predict 与手写前向最大差 %.2e\n', T{i,2}, class(net), T{i,3}, max(abs(z - zh)));
end

function W = getW(net)
    V = net.Learnables.Value;
    W.W1=undl(V{1}); W.b1=undl(V{2}); W.W2=undl(V{3}); W.b2=undl(V{4}); W.W3=undl(V{5}); W.b3=undl(V{6});
end

function v = undl(v)
    if isa(v,'dlarray'), v = extractdata(v); end
    v = double(v);
end
