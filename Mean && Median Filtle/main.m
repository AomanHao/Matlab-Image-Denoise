clear
close all
clc
%% 读取图像
I=imread('3096.jpg');
%% 彩色图像转灰度图像
if size(I,3) == 3
   I=rgb2gray(I);
else
end
figure;imshow(I);title('(a)原始图像');imwrite(I,'1.jpg');

%% 图像添加噪声
 I=imnoise(I,'salt & pepper',0.05); %加椒盐噪声
% I=imnoise(I,'gaussian',0,0.01); % 加高斯噪声
figure;imshow(I);title('(b)加噪图像');imwrite(I,'2.jpg');

[m,n]=size(I);
I=im2double(I);
r=3;%邻域窗大小
%% 运算时间统计
t=cputime;
tic;
%% 均值中值滤波
[I_mean,I_median]=compute_mean_median(I,r);
toc;
time_fcm_spatial_mean=cputime-t;%运算时间

figure;imshow(I_mean);title('均值滤波图像');imwrite(I,'2.jpg');
figure;imshow(I_median);title('中值滤波图像');imwrite(I,'3.jpg');

