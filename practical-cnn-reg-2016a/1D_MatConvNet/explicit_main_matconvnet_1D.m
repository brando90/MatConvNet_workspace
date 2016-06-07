clc;clear;clc;clear;
%% prepare Data
X_train = randn(1,1,1,1000);
x = X_train(:,:,:,1);
%%
L1 = 3;
L2 = 3;
%% Forward Pass
%
w1 = rand(1,1,1,L1);
%b1 = rand(L1,1);
b1 = [];
z1 = vl_nnconv(x,w1,b1) ;
a1 = vl_nnrelu(z1);
%
w2 = rand(1,1,1,L2);
%b1 = rand(L2,1);
b2 = [];
z2 = vl_nnconv(a1,w2,b2);
%%