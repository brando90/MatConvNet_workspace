clc;clear;clc;clear;
%% prepare Data
x = zeros(1,1,1,2);
x(:,1) = 1;
x(:,2) = 2;
%%
L1 = 3;
%% Forward Pass
w1 = zeros(1,1,1,L1); 
w1(:,:,:,1) = 1;
w1(:,:,:,2) = 3;
w1(:,:,:,3) = 5;
b1 = [];
z1 = vl_nnconv(x,w1,b1) 
disp('---');

G1 = ones(1,1,1,L1);
B1 = zeros(1,1,1,L1);
z1 = vl_nnbnorm(z1,G1,B1,'EPSILON',1e-10)

disp('---');
a1 = vl_nnrelu(z1) % (1 x 1 x 4)
disp('---');
w2 = zeros(1,1,1,L1); 
w2(:,:,:,1) = 1;
w2(:,:,:,2) = 2;
w2(:,:,:,3) = 3;
b2 = [];
z2 = vl_nnconv(a1,w2,b2) 
disp('---');
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
