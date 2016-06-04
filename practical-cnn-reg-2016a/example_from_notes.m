%% Data and Filter
% Create a random input image batch
x = randn(10, 10, 1, 2, 'single') ;
% Define a filter
w = single([
  0 -1 -0
  -1 4 -1
  0 -1 0]) ;
%% this is how forward and back mode look for convolution operator
y = vl_nnconv(x,w,b); % forward mode (get output)
p = randn(size(y), 'single'); % projection tensor (arbitrary)
% TODO: figure out what exactly do dx,dw,db mean.
[dx, dw, db] = vl_nnconv(x,w,b,p); % backward mode (get projected derivative)

%% this is how forward and back mode look for ReLu operator
y = vl_nnconv(x); % forward mode (get output)
p = randn(size(y), 'single'); % projection tensor (arbitrary)
% TODO: figure out what exactly do dx mean.
[dx] = vl_nnconv(x,p); % backward mode (get projected derivative)