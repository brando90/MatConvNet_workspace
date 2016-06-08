clc;clear;clc;clear;
%%
x = ones(1,1,1);
%%
[w1, w2] = two_weight_bank();
[ net ] = cnn_1D_init( 'batchNormalization', true, ...
    'w1', w1, 'w2', w2, ...
    'b1', [], 'b2', []);
%%
res = vl_simplenn(net, x ) ;
res(5).x