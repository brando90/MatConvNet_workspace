clc;clear;clc;clear;
%% l1 loss layer
x = randn(10, 10, 1, 2, 'single') ; 
r = randn(10, 10, 1, 2, 'single') ;
p = randn(1); %
%p = 1;

dl1dx = l1LossBackward(x,r,p);
%% check numerical
figure(22) ; clf('reset') ;
set(gcf, 'name', 'Part 2.3: check L1 layer') ;
func = @(X) l1LossForward(X,r); % l2LossForward(x,r)
checkDerivativeNumerically(func, x, dl1dx) ;