
% BFILTER2 Two dimensional bilateral filtering.
% This function implements 2-D bilateral filtering using B = bfilter2(A,W,SIGMA) performs 2-D bilateral filtering
% for the grayscale A. A should be a double precision matrix of size NxMx1 or NxMx3 (i.e., grayscale images,
% respectively) with normalized values in the closed interval [0,1]. The half-size of the Gaussian bilateral
% filter window is defined by W. The standard deviations of the bilateral filter are given by SIGMA,
% where the spatial-domain standard deviation is given by SIGMA(1) and the intensity-domain standard deviation is
% given by SIGMA(2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-process input and select appropriate filter.
function B = bfilter2(A,w,sigma)

% Verify that the input image exists and is valid.
if ~exist('A','var') || isempty(A)
   error('Input image A is undefined or invalid.');
end
if ~isfloat(A) || ~sum([1,3] == size(A,3)) || ...
      min(A(:)) < 0 || max(A(:)) > 1
   error(['Input image A must be a double precision ',...
          'matrix of size NxMx1 or NxMx3 on the closed ',...
          'interval [0,1].']);      
end

% Verify bilateral filter window size.
if ~exist('w','var') || isempty(w) || ...
      numel(w) ~= 1 || w < 1
   w = 5;
end
w = ceil(w);

% Verify bilateral filter standard deviations.
if ~exist('sigma','var') || isempty(sigma) || ...
      numel(sigma) ~= 2 || sigma(1) <= 0 || sigma(2) <= 0
   sigma = [3 0.1];
end

% Apply either grayscale or color bilateral filtering.
if size(A,3) == 1
   B = bfltGray(A,w,sigma(1),sigma(2));
else
   B = bfltColor(A,w,sigma(1),sigma(2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implements bilateral filtering for grayscale images.
function B = bfltGray(A,w,sigma_d,sigma_r)

% Pre-compute Gaussian distance weights.
[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2)/(sigma_d^2));
% G = exp(-(X.^2+Y.^2)/(2*sigma_d^2));

% Create waitbar.
h = waitbar(0,'Applying bilateral filter...');
set(h,'Name','Bilateral Filter Progress');

% Apply bilateral filter.
dim = size(A);
B = zeros(dim);
for i = 1:dim(1)
   for j = 1:dim(2)
      
         % Extract local region.
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         I = A(iMin:iMax,jMin:jMax);
      
         % Compute Gaussian intensity weights.
         H = exp(-(I-A(i,j)).^2/(2*sigma_r^2));
      
         % Calculate bilateral filter response.
         F = H.*G((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);
         B(i,j) = sum(F(:).*I(:))/sum(F(:));
               
   end
   waitbar(i/dim(1));
end

% Close waitbar.
close(h);



