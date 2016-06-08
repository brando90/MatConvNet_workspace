clc;clear;clc;clear;
%% start DAG CNN example
%run <MATCONVNETROOT>/matlab/vl_setupnn.m ; % activate MatConvNet if needed
net = dagnn.DagNN() ;
net2 = net ; % both net and net2 refer to the same object
%% the following command adds a layer an input x1, an output x2, and two parameters filters and biases.
convBlock = dagnn.Conv('size', [3 3 256 16], 'hasBias', true) ;
net.addLayer('conv1', convBlock, {'x1'}, {'x2'}, {'filters', 'biases'}) ;
%% ReLU layer
reluBlock = dagnn.ReLU() ;
net.addLayer('relu1', reluBlock, {'x2'}, {'x3'}, {}) ;
%% inspect net
net.layers(1); % shows the conents of block at layer 1
net.vars(1); % contents of variables
net.params(1); % contents of params
%% init
net.initParams(); % initialize the model parameters to random values
%% evaluate net
input = rand(10,15,256,1,'single');
net.eval({'x1',input}); % evaluates the network
%% recover output
i  = net.getVarIndex('x3');
output = net.vars(i).value; % net output
%% compute derivatives
proj_dzdy = rand(size(output), 'single'); % projection vector
net.eval({'x1',input},{'x3',proj_dzdy}); % compute derivatives
%% get derivatives
index_filter = net.getParamIndex('filters') ;
dzdfilters = net.vars(index_filter).der; % filter deriative