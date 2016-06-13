clc;clear;clc;clear;
%% prepare Data
M = 32; %batch size
X_train = zeros(1,1,1,M); % (1 1 1 2) = (1 1 1 M)
for m=1:M,
    X_train(:,:,:,m) = m; %training example value
end
Y_test = 10*X_train;
split = ones(1,M);
split(floor(M*0.75):end) = 2;
% load image dadabase (imgdb)
imdb.images.data = X_train;
imdb.images.label = Y_test;
imdb.images.set = split;
%% prepare parameters
L1=3;
w1 = randn(1,1,1,L1); %1st layer weights
w2 = randn(1,1,1,L1); %2nd layer weights
b1 = randn(1,1,1,L1); %1st layer biases
b2 = randn(1,1,1,L1); %2nd layer biases
G1 = ones(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN scale, one per  dimension
B1 = zeros(1,1,1,L1); % (1 1 1 3) = (1 1 1 L1) BN shift, one per  dimension
EPS = 1e-4;
%% make CNN layers: conv, BN, relu, conv, pdist, l2-loss
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...                          
                           'weights', {{w1, b1}}, ...
                           'learningRate', [0.9 0.9], ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'bnorm', ...
                           'weights', {{G1, B1}}, ...
                           'EPSILON', EPS, ...
                           'learningRate', [1 1 0.05], ...
                           'weightDecay', [0 0]) ;                       
net.layers{end+1} = struct('type', 'relu', ...
                           'name', 'relu1' ) ;
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv2', ...
                           'weights', {{w2, b2}}, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'pdist', ...
                           'name', 'averageing1', ...
                           'class', 0, ...
                           'p', 1) ;
%% add L2-loss                   
fwfun = @l2LossForward;
bwfun = @l2LossBackward;
net = addCustomLossLayer(net, fwfun, bwfun) ;
net.layers{end}.class = Y_test; % its the test set
net = vl_simplenn_tidy(net) ;
%% prepare train options
trainOpts.expDir = 'results/' ; %save results/trained cnn
trainOpts.gpus = [] ;
trainOpts.batchSize = 2 ;
trainOpts.learningRate = 0.02 ; %TODO: why is this learning rate here?
trainOpts.plotDiagnostics = false ;
%trainOpts.plotDiagnostics = true ; % Uncomment to plot diagnostics
trainOpts.numEpochs = 20 ; % number of training epochs
trainOpts.errorFunction = 'none' ;
%%
res = vl_simplenn(net, X_train, 1);
for epoch=1:trainOpts.numEpochs
    %% forward pass and compute derivatives
    projection = 1;
    res = vl_simplenn(net, X_train, projection);
    %% SGD
    num_layers = size(net, 2);
    for l=num_layers:-1:1
        for j=1:numel(res(l).dzdw)
            net.layers{l}.weights{j} = net.layers{l}.weights{j} - net.layers{l}.learningRate(j)*res(l).dzdw{j}; 
        end    
    end
end
%%
disp('END')
beep;