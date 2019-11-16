%% 程序分享 
% 西安邮电大学图像处理团队-郝浩
% 个人博客 www.aomanhao.top
% Github https://github.com/AomanHao
%--------------------------------------
clear
close all
clc
%% %%%%%%%%%%%%%%%图像%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I=imread('3096.jpg');

if size(I,3) == 3
   I=rgb2gray(I);
else
end
I=im2double(I);
figure;imshow(I);title('(a)原始图像');imwrite(I,'1.jpg');
% I=I;%不加噪声
% I=imnoise(I,'speckle',deta_2);
% I=imnoise(I,'salt & pepper',0.05); %加噪图
img1=imnoise(I,'gaussian',0,0.01); % 加高斯噪声

%% Set bilateral filter parameters.
w     = 5;       % bilateral filter half-width
sigma = [3 0.1]; % bilateral filter standard deviations

% Apply bilateral filter to each image.
bflt_img1 = jbfilter2(img1,img1,w,sigma);

% Display grayscale input image and filtered output.
figure(1); clf;
set(gcf,'Name','Grayscale Bilateral Filtering Results');
subplot(1,2,1); imagesc(img1);
axis image; colormap gray;
title('Input Image');
subplot(1,2,2); imagesc(bflt_img1);
axis image; colormap gray;
title('Result of Bilateral Filtering');