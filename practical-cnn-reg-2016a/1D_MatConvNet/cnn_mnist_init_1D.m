function [ output_args ] = cnn_mnist_init_1D( varargin )
opts.batchNormalization = true ;
opts.networkType = 'simplenn' ;
% vl_argparse fills up opts with arg_name=arg_val
opts = vl_argparse(opts, varargin) ;

net.layers = { } ;
net.layers{end+1} = struct(...
  'name', 'conv1', ...
  'type', 'conv', ...
  'weights', {{randn(1,1,1,L1,'single'), randn(L1,1,'single')}}, ...
  'pad', 0, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;
net.layers{end+1} = struct(...
  'name', 'relu1', ...
  'type', 'relu') ;
net.layers{end+1} = struct(...
  'name', 'output (conv2)', ...
  'type', 'conv', ...
  'weights', {{randn(1,1,1,L2,'single'), randn(1,1,'single')}}, ...
  'pad', 0, ...
  'stride', 1, ...
  'learningRate', [1 1], ...
  'weightDecay', [1 0]) ;
net.layers{end+1} = struct(...
  'name', 'L2 Loss', ...
  'type', 'pdist', ...
  'p', 2, ...
  'epsilon', lambda) ;
net = vl_simplenn_tidy(net) ; % Consolidate the network, fixing any missing option
end

