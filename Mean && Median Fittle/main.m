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
% I=imnoise(I,'salt & pepper',0.05); %加噪图
% I=imnoise(I,'gaussian',0,0.01); % 加高斯噪声

figure;imshow(I);title('(b)加噪图像');imwrite(I,'2.jpg');
[m,n]=size(I);

% I = I/255；
I=im2double(I);
[I_mean,I_median]=compute_mean_median(I,r);%扩展邻域
% I_median=double(I_median);


I4 = I(:);  %% 将图像灰度按列排列
X_spatial_mean = I_mean(:);
% X_spatial_median = I_median(:);

%% ------------------------ fcm_spatial_mean------------------------
fcm_spatial_mean_label=zeros(m*n,1);
t=cputime;
tic;

toc;
time_fcm_spatial_mean=cputime-t;

figure;imshow(I);title('(a)原始图像');imwrite(I,'1.jpg');
