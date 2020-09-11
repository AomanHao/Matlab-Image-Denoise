%% 程序分享 
% 个人博客 www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
%--------------------------------------


clear
close all
clc
%% 读取图像
I=imread('3096.jpg');

if size(I,3) == 3
   I=rgb2gray(I);
else
end
I=im2double(I);
figure;imshow(I);title('(a)原始图像')
%% 添加噪声
%I=imnoise(I,'speckle',deta_2);
% I=imnoise(I,'salt & pepper',0.05); %加噪图
I_noise=imnoise(I,'gaussian',0,0.01); % 加高斯噪声s
figure;imshow(I_noise);title('(b)加噪图像');

%% 形态学滤波
se=3; % the parameter of structuing element used for morphological reconstruction
data=w_recons_CO(I_noise,strel('square',se));

figure;imshow(data);title('去噪图像');

