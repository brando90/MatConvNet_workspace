function dx = l1LossBackward(x,r,p)
delta = x - r;
delta( delta < 0 ) = -1;
delta( delta >= 0 ) = 1;
dx = p*delta;
dx = dx / (size(x,1) * size(x,2)) ;  % normalize by image size