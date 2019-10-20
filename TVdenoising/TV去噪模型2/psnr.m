%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% >>>> IMAGEBOX >>>> JX >>>> UCLA >>>>
%
% image toolbox
% MATLAB file
% 
% snr.m
% compute  peak signal-to-noise-ratio (SNR) of a noisy signal/image
%
% function s = psnr(noisydata, original)
%
% input:  noisydata: noisy data
%         original:  clean data
% output: s: PSNR value
% example: s = snr(f, eu);
%
% created:       03/15/2008
% last modified: 12/03/2005
% author:        lubibo@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function s = psnr(noisydata, original)

noisydata=double(noisydata);
original=double(original);

[m,n] = size(noisydata);

peak=255*255*m*n;

noise  = noisydata - original;
nostotal = sum(sum(noise.*noise));

if nostotal == 0
    s = 999.99; %% INF. clean image
else
    s = 10 * log10(peak./nostotal);
end

return
