% JBFILTER2 Two dimensional Joint bilateral filtering.
%    This function implements 2-D bilateral filtering using
%    the method outlined in, however with weights calculated according
%    to another image.  
%
%    B = jbfilter2(D,C,W,SIGMA) performs 2-D bilateral filtering
%    for the grayscale or color image A. D should be a double
%    precision matrix of size NxMx1 (i.e., grayscale) 
%    with normalized values in the closed interval [0,1]. 
%    C should be similar to D, from which the weights are 
%    calculated, with normalized values in the closed 
%    interval [0,1].  The half-size of the Gaussian
%    bilateral filter window is defined by W. The standard
%    deviations of the bilateral filter are given by SIGMA,
%    where the spatial-domain standard deviation is given by
%    SIGMA(1) and the intensity-domain standard deviation is
%    given by SIGMA(2).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-process input and select appropriate filter.
function B = jbfilter2(D,C,w,sigma)

% Verify that the input image exists and is valid.
if ~exist('D','var') || isempty(D)
   error('Input image D is undefined or invalid.');
end
if ~isfloat(D) || ~sum([1,3] == size(D,3)) || ...
      min(D(:)) < 0 || max(D(:)) > 1
   error(['Input image D must be a double precision ',...
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
if size(D,3) == 1
   B = jbfltGray(D,C,w,sigma(1),sigma(2));
else
   B = jbfltGray(D,C,w,sigma(1),sigma(2));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implements bilateral filtering for grayscale images.
function B = jbfltGray(D,C,w,sigma_d,sigma_r)

% Pre-compute Gaussian distance weights.
[X,Y] = meshgrid(-w:w,-w:w);
G = exp(-(X.^2+Y.^2)/(2*sigma_d^2));

% Create waitbar.
%h = waitbar(0,'Applying bilateral filter on gray image...');
%set(h,'Name','Bilateral Filter Progress');

% Apply bilateral filter.
dim = size(D);
B = zeros(dim);
for i = 1:dim(1)
   for j = 1:dim(2)
      
         % Extract local region.
         iMin = max(i-w,1);
         iMax = min(i+w,dim(1));
         jMin = max(j-w,1);
         jMax = min(j+w,dim(2));
         I = D(iMin:iMax,jMin:jMax);
         
         % To compute weights from the color image
         J = C(iMin:iMax,jMin:jMax);
      
         % Compute Gaussian intensity weights according to the color image
         H = exp(-(J-C(i,j)).^2/(2*sigma_r^2));
      
         % Calculate bilateral filter response.
         F = H.*G((iMin:iMax)-i+w+1,(jMin:jMax)-j+w+1);
         B(i,j) = sum(F(:).*I(:))/sum(F(:));
               
   end
   %waitbar(i/dim(1));
end

% Close waitbar.
%close(h);
