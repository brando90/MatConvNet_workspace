% install and compile MatConvNet (needed once)
disp('install and compile MatConvNet (needed once)');
untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta20.tar.gz') ;
cd matconvnet-1.0-beta20
run matlab/vl_compilenn

% download a pre-trained CNN from the web (needed once)
disp('download a pre-trained CNN from the web (needed once)');
urlwrite(...
  'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
  'imagenet-vgg-f.mat') ;

% setup MatConvNet
disp('setup MatConvNet');
run  matlab/vl_setupnn

% load the pre-trained CNN
disp('load the pre-trained CNN');
net = load('imagenet-vgg-f.mat') ;
net = vl_simplenn_tidy(net) ; % Fix an incomplete or outdated SimpleNN network. takes the NET object and upgrades it to the current version of MatConvNet.

% load and preprocess an image
disp('load and preprocess an image');
im = imread('peppers.png') ;
im_ = single(im) ; % note: 0-255 range
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

% run the CNN
disp('run the CNN');
res = vl_simplenn(net, im_) ;

% show the classification result
disp('show the classification result');
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;