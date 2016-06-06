clc;clear;clc;clear;
%% Eucledian distance layer
x = randn(10, 10, 1, 2, 'single') ; % fake data
r = randn(10, 10, 1, 2, 'single') ; % reference tensor
p = randn(1); % projection

%% compute analytic derivative
dl2dx = l2LossBackward(x,r,p); % computes analytic derivative = backard pass
%% figures for difference between analytic and numerical
figure(22) ; clf('reset') ;
set(gcf, 'name', 'Part 2.3: check L2 layer') ;
%% compute numerical derivative
func = @(X) l2LossForward(X,r); % l2 loss = forward pass
checkDerivativeNumerically(func, x, dl2dx) ;