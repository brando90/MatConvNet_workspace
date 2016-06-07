clc;clear;clc;clear;
%%
net.layers{1} = struct(...
    'name', 'conv1', ...
    'type', 'conv', ...
    'weights', {{randn(10,10,3,2,'single'), randn(2,1,'single')}}, ...
    'pad', 0, ...
    'stride', 1) ;
net.layers{2} = struct(...
    'name', 'relu1', ...
    'type', 'relu') ;
%%
data = randn(300, 500, 3, 5, 'single') ;
net = vl_simplenn_tidy(net) ; 
%% evaluate CNN
res = vl_simplenn(net, data) ;