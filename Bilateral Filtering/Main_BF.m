%% 程序分享 
% 个人博客 www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
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

%% 设置参数
w     = 5;       % bilateral filter half-width
sigma = [3 0.1]; % bilateral filter standard deviations

%% 双边滤波
bflt_img1 = bfilter2(img1,w,sigma);

%% 输出图像
figure;imshow(I); title('Input Image');
figure;imshow(img1); title('Noise Image');
figure;imshow(bflt_img1);title('Result of Bilateral Filtering');