clc;clear;clc;clear;
%% prepare Data
M = 3;
x = zeros(1,1,1,M); % (1 1 1 2) = (1 1 1 M)
for m=1:M,
    x(:,:,:,m) = m;
end
%%
L1 = 3;
%% Forward Pass
w1 = zeros(1,1,1,L1); % (1 1 1 L1) = (1 1 1 3)
w1(:,:,:,1) = 1;
w1(:,:,:,2) = 2;
w1(:,:,:,3) = 3;
b1 = [];
z1 = vl_nnconv(x,w1,b1); % (1 1 3 2) = (1 1 L1 M)
Z1 = squeeze(z1);
disp('---');

% BN scale, one per  dimension
G1 = ones(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1)
% BN shift, one per  dimension
B1 = zeros(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1)
b1 = vl_nnbnorm(z1,G1,B1,'EPSILON',1e-10); % (1 1 3 2) = (1 1 L1 M)
B1 = squeeze(b1)
%b1 = z1;

% k=1;
% m=1;
% ( Z1(k,m) - mean(Z1(k,:)) ) / (std(Z1(k,:)) + 1e-10)
% ( z1(1,1,k,m) - mean(z1(1,1,k,:)) ) / sqrt(var(z1(1,1,k,:)) + 1e-10)

disp('---');
a1 = vl_nnrelu(b1); % (1 1 3 2) = (1 1 L1 M) 
disp('---');
w2 = zeros(1,1,1,L1); % (1 1 1 L1) = (1 1 1 3)
w2(:,:,:,1) = 1;
w2(:,:,:,2) = 2;
w2(:,:,:,3) = 3;
b2 = [];
z2 = vl_nnconv(a1,w2,b2);
disp('---');
y1 = vl_nnpdist(z2, 0, 1);
%%
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w1, []}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                            'weights', {{ones(1,1,1,L1), zeros(1,1,1,L1)}}, ...
                            'learningRate', [1 1 0.05], ...
                            'weightDecay', [0 0]) ;                       
net.layers{end+1} = struct('type', 'relu', ...
                           'name', 'relu1' ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w2, []}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;
net = vl_simplenn_tidy(net) ;
res = vl_simplenn(net, x);
disp('---');
y1
disp('---');
y2 = res(end).x
%%
