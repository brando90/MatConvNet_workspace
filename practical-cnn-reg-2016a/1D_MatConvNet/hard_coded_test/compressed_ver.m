clc;clear;clc;clear;
%% prepare Data
M = 5;
x = randn(1,1,1,M); % (1 1 1 2) = (1 1 1 M)
%%
L1 = 3;
%% Forward Pass
w1 = randn(1,1,1,L1); % (1 1 1 L1) = (1 1 1 3)
w2 = randn(1,1,1,L1);
z1 = vl_nnconv(x,w1,[]); % (1 1 3 2) = (1 1 L1 M)

% BN scale, one per  dimension
G1 = randn(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1)
% BN shift, one per  dimension
B1 = randn(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1)
b1 = vl_nnbnorm(z1,G1,B1,'EPSILON',1e-10); % (1 1 3 2) = (1 1 L1 M)
a1 = vl_nnrelu(b1); % (1 1 3 2) = (1 1 L1 M) 
z2 = vl_nnconv(a1,w2,[]) 
y1 = vl_nnpdist(z2, 0, 1)

%%
% net.layers = {} ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'name', 'conv1', ...
%                            'weights', {{w1, []}}, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu', ...
%                            'name', 'relu1' ) ;
% net.layers{end+1} = struct('type', 'conv', ...
%                            'name', 'conv1', ...
%                            'weights', {{w2, []}}, ...
%                            'pad', 0) ;
% net.layers{end+1} = struct('type', 'pdist', ...
%                            'name', 'averageing1', ...
%                            'class', 0, ...
%                            'p', 1) ;
% net = vl_simplenn_tidy(net) ;
% res = vl_simplenn(net, x);
% y1
% y2 = res(end).x
%%
