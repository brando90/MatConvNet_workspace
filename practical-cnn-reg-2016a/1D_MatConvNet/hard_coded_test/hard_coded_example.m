clc;clear;clc;clear;
%% prepare Data
x = zeros(1,1,1); % (1 1 1)
x(:,1) = 1;
%%
L1 = 3;
%% Init weights
% w1
w1 = zeros(1,1,1,L1); % (1 1 1 L1) = (1 1 1 3)
w1(:,:,:,1) = 1;
w1(:,:,:,2) = 2;
w1(:,:,:,3) = 3;
%b1 = [];
b1 = ones(1,L1);
% w2
w2 = zeros(1,1,1,L1); % (1 1 1 L1) =(1 1 1 3)
w2(:,:,:,1) = 1;
w2(:,:,:,2) = 2;
w2(:,:,:,3) = 3;
%b2 = [];
b2 = ones(1,L1);
%% Forward Pass
z1 = vl_nnconv(x,w1,b1) ; % (1 1 L1) = (1 1 3)
disp('---');
a1 = vl_nnrelu(z1) % (1 1 L1) = (1 1 3)
disp('---');
z2 = vl_nnconv(a1,w2,b2) % (1 1 L1) = (1 1 3)
disp('---');
y1 = vl_nnpdist(z2, 0, 1); % (1 1)
%%
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w1, b1}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu', ...
                           'name', 'relu1' ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w2, b1}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;
net = vl_simplenn_tidy(net) ;
res = vl_simplenn(net, x);
y1 % (1 1)
y2 = res(end).x % (1 1)
%%
