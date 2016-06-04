%% 1 pixel image
D = 1;
x = rand(D,D)

%w = [10];
%y = vl_nnconv(x, w, []) 

w1 = single([1]) ;

w2 = single([-1]) ;

w3 = single([10]) ;

wbank = cat(4, w1, w2, w3) ;

y = vl_nnconv(x, wbank, []) 

%%