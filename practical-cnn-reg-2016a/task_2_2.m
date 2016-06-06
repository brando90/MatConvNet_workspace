%% Part 2.2: Backpropagation
clc;clear;clc;clear;
% Create a random input image batch
x = randn(10, 10, 1, 2, 'single') ;

% Define a filter
w1 = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

w2 = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;

% Forward mode: evaluate the conv follwed by ReLU
z1 = vl_nnconv(x, w1, []) ;
z2 = vl_nnrelu(z1) ;
z3 = vl_nnconv(z2, w2, []) ;

% Pick a random projection tensor
p3 = randn(size(z3), 'single') ;

% Backward mode: projected derivatives
% d<p,z3>/dw1 and % d<p,z3>/dw2
[dz3dz2, dz3dw2] = vl_nnconv(z2, w2, [], p3) ;
dz2dz1 = vl_nnrelu(z1, dz3dz2) ;
[dz3dx,dz3w1] = vl_nnconv(x, w1, [], dz2dz1) ;


% Check the derivative numerically
figure(22) ; clf('reset') ;
set(gcf, 'name', 'Task Part 2.2: three layers backrpop') ;
func = @(W) proj( vl_nnconv(vl_nnrelu( vl_nnconv(x,W,[]) ) ,w2,[]), p3) ;
checkDerivativeNumerically(func, w1, dz3w1) ;
%func = @(W) proj( vl_nnconv(vl_nnrelu( vl_nnconv(x,w1,[]) ) ,W,[]), p3) ;
%checkDerivativeNumerically(func, w2, dz3dw2) ;