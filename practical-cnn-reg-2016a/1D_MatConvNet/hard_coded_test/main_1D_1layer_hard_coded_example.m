clc;clear;clc;clear;
%% prepare Data
M = 6;
X = zeros(1,1,1,M); % (1 1 1 2) = (1 1 1 M)
for m=1:M,
    X(:,:,:,m) = m;
end
split = ones(1,M);
split(floor(M*0.75):end) = 2;
imdb.images.data = X;
%imdb.images.label = squeeze( X );
imdb.images.label = X;
imdb.images.set = split;
%%
L1=3;
w1 = randn(1,1,1,L1);
w2 = randn(1,1,1,L1);
b1 = randn(1,L1);
b2 = randn(1,L1);
%%
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w1, b1}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{ones(1,1,1,L1), zeros(1,1,1,L1)}}, ...
                           'learningRate', [1 1 0.05], ...
                           'weightDecay', [0 0]) ;                       
net.layers{end+1} = struct('type', 'relu', ...
                           'name', 'relu1' ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{w2, b2}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;                    
%% loss layer
fwfun = @l2LossForward;
bwfun = @l2LossBackward;
net = addCustomLossLayer(net, fwfun, bwfun) ;
net = vl_simplenn_tidy(net);
%% Train params
trainOpts.expDir = 'results/' ;
trainOpts.gpus = [] ;
trainOpts.batchSize = 16 ;
trainOpts.learningRate = 0.02 ;
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 20 ;
trainOpts.errorFunction = 'none' ;
%% CNN TRAIN
net = cnn_train(net, imdb, @getBatch, trainOpts) ;