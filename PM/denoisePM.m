function [sol, resimg] = denoisePM(img, params, edgestopfct, startsol, precision)
% DENOISEPM Denoise an image with Perona-Malik isotropic diffusion
% [SOL, RESIMG] = denoisePM(IMG, PARAMS, EDGESTOPFCT, STARTSOL)
% SOL: Solution. The diffused image.
% RESIMG: Optional residual image.
% IMG: 2D image matrix
% PARAMS:   Parameter struct:
%   In this function, the following parameters are used:
%   DENOISEPM_SIGMA: standard derivation value/list of the edge-stopping 
%   function 
%   DENOISEPM_TIME: Time parameter - amount of diffusion applied.
%   DENOISEPM_SMOOTH: standart derivation of pre-gradient-computation 
%   smoothing (suggestion: something small, for example 0.5)
%   DENOISEPM_MAXITER: Maximum number of iterations. (suggestion: 1000)
% EDGESTOPFCT: edge-stopping function used. Possibilities are: 'perona',
% 'tukey', 'tukeylog', 'complex'. Default: tukey
% STARTSOL: Initialisation for the solution, by default the input image
%
% Calling examples:
% 
% First Example: 0..255 scaled intensity values
% params.DENOISEPM_SIGMA = 20;
% params.DENOISEPM_TIME = 3;
% params.DENOISEPM_SMOOTH = 0;
% params.DENOISEPM_MAXITER = 1000;
% sol = denoisePM(img, params);  
%
% Second example: 0..255 scaled intensity values
% params.DENOISEPM_SIGMA = [20 pi/1000];
% params.DENOISEPM_TIME = 3;
% params.DENOISEPM_SMOOTH = 1;
% params.DENOISEPM_MAXITER = 200;
% sol = denoisePM(img, params, 'complex'); 
%
% A list of cites for the different edgestopfunctions:
%
% 'perona'
% P. Perona, J. Malik:	Scale-Space and Edge Detection Using Anisotropic
% Diffusion, IEEE Trans. Pattern Anal. Mach. Intell., 
% 12(7), 1990, 629?639.
%
% 'tuckey'
% M. J. Black, G. Sapiro, D. Marimont, D. Heeger: Robust anisotropic
% diffusion, IEEE Trans. on Image Processing, 
% 7(3), 1998, 421?432.
%
% 'complex'
% G. Gilboa, N. A. Sochen, Y.Y. Zeevi, ?Image Enhancement and Denoising by 
% Complex Diffusion Processes?, 
% IEEE Trans. Pattern Anal. Mach. Intell. 26(8), 1020?1036 (2004).
%
% The corresponding 'log' version compute their coefficients on a 
% logarithmic version of the image.
%
% The lagged diffusitivy solution of the PDE was proposed in:
% C. R. Vogel, M. E. Oman: Iterative Methods for Total Variation Denoising,
% SIAM Journal on Scientific Computing, 17(1), 1996, 227?238.
% 
% Further discussion on the method can be found in:
% T. Chan, P. Mulet: On the convergence of the lagged diffusivity fixed
% point method in total variation image restoration,
% SIAM journal on numerical analysis, 36(2), 1999, 354?367.
%
% Implementation by Markus Mayer, Pattern Recognition Lab, 
% University of Erlangen-Nuremberg, 2008
% This version was revised and commented in August 2010
%
% You may use this code as you want. I would be grateful if you would go to
% my homepage look for articles that you find worth citing in your next
% publication:
% http://www5.informatik.uni-erlangen.de/en/our-team/mayer-markus
% Thanks, Markus

if nargin < 2
    error('Not enough input arguments.');
end
if nargin < 3
    edgestopfct = 'tukey';
end
if nargin < 5
    presicion = 'single';
end

sigma = params.DENOISEPM_SIGMA;
t = params.DENOISEPM_TIME;

% Add a 1-border to the image
img = [img(:,1), img, img(:,size(img,2))];
img = vertcat(img(1,:), img, img(size(img,1),:));

iter = 0; % iteration counter

if nargin < 4
    sol = img; % solution initialisation: original image
else
    sol = [startsol(:,1), startsol, startsol(:,size(startsol,2))];
    sol = vertcat(sol(1,:), sol, sol(size(sol,1),:));
end

stencilN = zeros(size(img, 1), size(img, 2), presicion);
stencilS = zeros(size(img, 1), size(img, 2), presicion);
stencilE = zeros(size(img, 1), size(img, 2), presicion);
stencilW = zeros(size(img, 1), size(img, 2), presicion);
stencilM = zeros(size(img, 1), size(img, 2), presicion);

resimg = img;
resold = 1e+20; % old residual
resarr = [1e+20 1e+20 1e+20]; % array of the last 3 residual changes

% diffusion iteration stoping criteria
% if the sum of the last 5 residuals is negative, a completion of the
% calculation is assumed.
% Algorithm: No explicit time marching is applied, instead the linear
% equation system the diffusion process creates for one timestep is solved
% with lagged diffusivity by R/B Gauss-Seidel Iteration steps
if params.DENOISEPM_SMOOTH ~= 0
    gauss1 = fspecial('gaussian', ...
        round(params.DENOISEPM_SMOOTH * params.DENOISEPM_SMOOTH + 1), ...
        params.DENOISEPM_SMOOTH);
end

while (sum(resarr) > 0) && (iter < params.DENOISEPM_MAXITER);
    % Calculation of the edge-stoping function

    if params.DENOISEPM_SMOOTH ~= 0
        smoothsol = imfilter(sol, gauss1, 'symmetric');

        if strcmp(edgestopfct,  'tukey')
            coeff = tukeyEdgeStop(smoothsol, sigma);
        elseif strcmp(edgestopfct, 'perona')
            coeff = peronaEdgeStop(smoothsol, sigma);
        elseif strcmp(edgestopfct, 'tukeylog')
            coeff = tukeyEdgeStopLog(smoothsol, sigma);
        elseif strcmp(edgestopfct, 'complex')
            coeff = complexEdgeStop(smoothsol, sigma);
        else
            error('Give a suitable edgestopfct argument');
        end
    else
        if strcmp(edgestopfct,  'tukey')
            coeff = tukeyEdgeStop(sol, sigma);
        elseif strcmp(edgestopfct, 'perona')
            coeff = peronaEdgeStop(sol, sigma);
        elseif strcmp(edgestopfct, 'tukeylog')
            coeff = tukeyEdgeStopLog(sol, sigma);
        elseif strcmp(edgestopfct, 'complex')
            coeff = complexEdgeStop(sol, sigma);
        else
            error('Give a suitable edgestopfct argument');
        end
    end

    coeff = coeff * t;

    % stencil computation
    stencilN(2:end-1, 2:end-1) = (coeff(2:end-1, 2:end-1) + coeff(1:end-2, 2:end-1))/2;
    stencilS(2:end-1, 2:end-1) = (coeff(2:end-1, 2:end-1) + coeff(3:end, 2:end-1))/2;
    stencilE(2:end-1, 2:end-1) = (coeff(2:end-1, 2:end-1) + coeff(2:end-1, 3:end))/2;
    stencilW(2:end-1, 2:end-1) = (coeff(2:end-1, 2:end-1) + coeff(2:end-1, 1:end-2))/2;

    stencilM = stencilN + stencilS + stencilE + stencilW + 1;

    % solution computation: R/B Gauss Seidel
    sol(2:2:end-1, 2:2:end-1) = (img(2:2:end-1, 2:2:end-1) ...
        + (stencilN(2:2:end-1, 2:2:end-1) .* sol(1:2:end-2, 2:2:end-1) ...
        + stencilS(2:2:end-1, 2:2:end-1) .* sol(3:2:end, 2:2:end-1) ...
        + stencilE(2:2:end-1, 2:2:end-1) .* sol(2:2:end-1, 3:2:end)...
        + stencilW(2:2:end-1, 2:2:end-1) .* sol(2:2:end-1, 1:2:end-2))) ...
        ./ stencilM(2:2:end-1, 2:2:end-1);

    sol(3:2:end, 3:2:end) = (img(3:2:end, 3:2:end) ...
        + (stencilN(3:2:end, 3:2:end) .* sol(2:2:end-1, 3:2:end) ...
        + stencilS(3:2:end, 3:2:end) .* sol(4:2:end, 3:2:end) ...
        + stencilE(3:2:end, 3:2:end) .* sol(3:2:end, 4:2:end) ...
        + stencilW(3:2:end, 3:2:end) .* sol(3:2:end, 2:2:end-1))) ...
        ./ stencilM(3:2:end, 3:2:end);

    sol(2:2:end-1, 3:2:end) = (img(2:2:end-1, 3:2:end) ...
        + (stencilN(2:2:end-1, 3:2:end) .* sol(1:2:end-2, 3:2:end) ...
        + stencilS(2:2:end-1, 3:2:end) .* sol(3:2:end, 3:2:end) ...
        + stencilE(2:2:end-1, 3:2:end) .* sol(2:2:end-1, 4:2:end) ...
        + stencilW(2:2:end-1, 3:2:end) .* sol(2:2:end-1, 2:2:end-1))) ...
        ./ stencilM(2:2:end-1, 3:2:end);

    sol(3:2:end, 2:2:end-1) = (img(3:2:end, 2:2:end-1) ...
        + (stencilN(3:2:end, 2:2:end-1) .* sol(2:2:end-1, 2:2:end-1) ...
        + stencilS(3:2:end, 2:2:end-1) .* sol(4:2:end, 2:2:end-1) ...
        + stencilE(3:2:end, 2:2:end-1) .* sol(3:2:end, 3:2:end) ...
        + stencilW(3:2:end, 2:2:end-1) .* sol(3:2:end, 1:2:end-2))) ...
        ./ stencilM(3:2:end, 2:2:end-1);

    % residual computation
    resimg(2:end-1, 2:end-1) = (-(stencilN(2:end-1, 2:end-1) .* sol(1:end-2, 2:end-1) ...
        + stencilS(2:end-1, 2:end-1) .* sol(3:end , 2:end-1) ...
        + stencilE(2:end-1, 2:end-1) .* sol(2:end-1, 3:end) ...
        + stencilW(2:end-1, 2:end-1) .* sol(2:end-1, 1:end-2))) ...
        + stencilM(2:end-1, 2:end-1) .* sol(2:end-1, 2:end-1) - img(2:end-1, 2:end-1);

    res = sum(sum(real(resimg) .* real(resimg)));

    resdiff = resold - res;
    resold = res;
    resarr = [resdiff resarr(1, 1:(size(resarr, 2)-1))];

    % duplicate edges as new borders
    sol = [sol(:,2), sol(:, 2:end-1), sol(:,end-1)];
    sol = vertcat(sol(2,:), sol(2:end-1, :), sol(end-1,:));
    
    % disp(resarr);
    iter = iter + 1;
end

% Remove border
sol = sol(2:(size(sol,1)-1), 2:(size(sol,2)-1));
if(nargout > 1)
    resimg = resimg(2:(size(sol,1)-1), 2:(size(sol,2)-1));
end;

%--------------------------------------------------------------------------

function r = tukeyEdgeStop(img, sigma)
% TUKEYEDGESTOP Tukey Edge-Stopping function on matrix
% tukeyEdgeStop(img, sigma)
% img: 2D image matrix
% sigma: standard derivation parameter

r = gradientAbsolute(img);
r = 1 - r .* r / (sigma * sigma);
r(r < 0) = 0;
r = r .* r;

%--------------------------------------------------------------------------

function r = tukeyEdgeStopLog(img, sigma)
% TUKEYEDGESTOPLOG Tukey Edge-Stopping function on matrix, added log
% tukeyEdgeStopLog(img, sigma)
% img: 2D image matrix
% sigma: standard derivation parameter

r = log(img);
r = gradientAbsolute(r);
r = 1 - r .* r / (sigma * sigma);
r(r < 0) = 0;
r = r .* r;

%--------------------------------------------------------------------------

function r = peronaEdgeStop(img, sigma)
% PERONAEDGESTOP Perona Edge-Stopping function on matrix
% peronaEdgeStop(img, sigma)
% img: 2D image matrix
% sigma: standard derivation parameter

r = gradientAbsolute(img);
r = exp(- r .* r / (2 * sigma * sigma));

%--------------------------------------------------------------------------

function r = complexEdgeStop(img, sigma)
% COMPLEXEDGESTOP Complex edge-stopping function by Gilboa
% complexEdgeStop(img, sigma)
% img: 2D image matrix, can be complex
% sigma (2)-Vector, first entry: Diffusion stopper, second: theta

j = sqrt(-1);
r = gradientAbsolute(img);
r = exp(j * sigma(2))./(1+(imag(r) / (sigma(1) * sigma(2))) .^ 2 );

%  r = exp(j .* sigma(2))./(1+(imag(r) ./ (sigma(1) .* angle(img))) .^ 2 );
% r = exp(j .* angle(img))./(1+(imag(r) ./ (sigma(1) .* angle(img))) .^ 2 );

%--------------------------------------------------------------------------

function r = gradientAbsolute(A)
% GRADIENTABSOLUTE Returns the absolute value of the gradient
% gradientAbsolute(A): A must be a 2D matrix

[Gx, Gy] = gradient(A);
r = sqrt(Gx .* Gx + Gy .* Gy);


