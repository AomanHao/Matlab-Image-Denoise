%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% >>>> IMAGEBOX >>>> JX >>>> UCLA >>>>
%
% image toolbox
% MATLAB file
% 
% rmse.m
% compute  root-mean-square-error (RMSE) of a noisy signal/image
%
% function s = rmse(noisydata, original)
%
% input:  noisydata: noisy data
%         original:  clean data
% output: s: RMSE value
% example: s = rmse(f, eu);
%
% created:       08/23/2009
% last modified: 12/03/2005
% author:        gzc@email.jlu.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function E = mae(noisydata, original)

noisydata=double(noisydata);
original=double(original);

[m,n] = size(noisydata);


noise  = abs(noisydata - original);
nostotal = sum(sum(noise));

E=nostotal/(m*n);

return
