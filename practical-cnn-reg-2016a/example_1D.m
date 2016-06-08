clc;clear;clc;clear;
%% prepare Data
x = zeros(1,1,1);
x(:,1) = 1;
%%
L1 = 3;
%% Forward Pass
w1 = zeros(1,1,1,L1); 
w1(:,:,:,1) = 1;
w1(:,:,:,2) = 2;
w1(:,:,:,3) = 3;
b1 = [];
z1 = vl_nnconv(x,w1,b1) ;
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
a2 = vl_nnrelu(z2) % want (1 x 1 x 1)
%%