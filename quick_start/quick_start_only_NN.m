%% load the pre-trained CNN
disp('load the pre-trained CNN');
net = load('imagenet-vgg-f.mat') ;
% Fix an incomplete or outdated SimpleNN network. takes the NET object and upgrades it to the current version of MatConvNet.
net = vl_simplenn_tidy(net) ; %takes the NET object and upgrades it to the current version of MatConvNet.

%% load and preprocess an image
disp('load and preprocess an image');
im = imread('peppers.png') ; % 384 x 512 x 3 (uint8)
%converts the vector X to single precision. X can be any numeric object (such as a DOUBLE)
im_ = single(im) ; % note: 0-255 range (
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
% centers image
im_ = bsxfun(@minus, im_, net.meta.normalization.averageImage) ;

%% run the CNN
disp('run the CNN');
% evaluates the convnet NET on data X.
res = vl_simplenn(net, im_) ;

%% show the classification result
disp('show the classification result');
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;