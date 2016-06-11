clc;clear;clc;clear;
%% prepare Data
M = 3;
x = zeros(1,1,1,M); % (1 1 1 2) = (1 1 1 M)
for m=1:M,
    x(:,:,:,m) = m;
end
Y = 5;
r=Y;
%% parameters
L1 = 3;
w1 = randn(1,1,1,L1); % (1 1 1 L1) = (1 1 1 3)
b1 = ones(1,L1);
w2 = randn(1,1,1,L1); % (1 1 1 L1) = (1 1 1 3)
b2 = ones(1,L1);
%% Forward Pass
z1 = vl_nnconv(x,w1,b1); % (1 1 3 2) = (1 1 L1 M)
G1 = ones(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN scale, one per  dimension
B1 = zeros(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN shift, one per  dimension
bn1 = vl_nnbnorm(z1,G1,B1,'EPSILON',1e-10); % (1 1 3 2) = (1 1 L1 M)
a1 = vl_nnrelu(bn1); % (1 1 3 2) = (1 1 L1 M) 
z2 = vl_nnconv(a1,w2,b2);
y1 = vl_nnpdist(z2, 0, 1);
loss_forward = l2LossForward(x,r);
%%
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w1, b1}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                            'weights', {{w1, b1}}, ...
                            'learningRate', [1 1 0.05], ...
                            'weightDecay', [0 0]) ;                       
net.layers{end+1} = struct('type', 'relu', ...
                           'name', 'relu1' ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w2, b2}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;
fwfun = @l2LossForward;
bwfun = @l2LossBackward;
net = addCustomLossLayer(net, fwfun, bwfun) ;
net.layers{end}.class = Y;
net = vl_simplenn_tidy(net) ;
res = vl_simplenn(net, x);
%%
loss_forward = squeeze( loss_forward ) % (1 1)
loss_res = squeeze( res(end).x ) % (1 1)
%% Backward Pass
p = 1;
dldx = l2LossBackward(y1,r,p);
dy1dx = vl_nnpdist(z2, 0, 1, dldx);
[dz2dx, dz2dw2] = vl_nnconv(a1, w2, b2, dy1dx);
da1dx = vl_nnrelu(bn1, dz2dx);
[dbn1dx,dbn1dG1,dbn1dB1] = vl_nnbnorm(z1,G1,B1,da1dx);
[dz1dx, dz1dw1] = vl_nnconv(x, w1, b1, dbn1dx);
%%
dzdy = 1;
res = vl_simplenn(net, x, dzdy, res);
%%
dz1dx = squeeze(dz1dx)
vl_simplenn_dz1dx = squeeze(res(1).dzdx)