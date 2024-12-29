%% 程序分享
% 个人博客 www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
%--------------------------------------
clear
close all
clc
%% %%%%%%%%%%%%%%% load data %%%%%%%%%%%%%%%%%
addpath('./methods/');
I=imread('3096.jpg');

if size(I,3) == 3
    I=rgb2gray(I);
else
end
I=im2double(I);
figure;imshow(I);title('(a)原始图像');


% I=I;%不加噪声
% I=imnoise(I,'speckle',deta_2);
% I=imnoise(I,'salt & pepper',0.05); %加噪图
I=imnoise(I,'gaussian',0,0.001); % 加高斯噪声
imwrite(I,strcat('./result/','re','.png'));

%% 设置参数
w     = 1;       % bilateral filter half-width

tic;
method = 'bfilter2';
switch method
    case 'bfilter2'
        %% 双边滤波
        sigma = [2 0.1]; % bilateral filter standard deviations
        result = bfilter2(I,w,sigma);
        
    case 'Joint_bfilter2'
        %% 联合双边滤波
        sigma = [2, 0.1, 0.5, 0.5]; 
        conf.BF_type = 'BiF';%BiF  SiF
        result = fun_Joint_bfilter2(I,w,sigma);
        
end
toc;
t=toc;
imwrite(double(result),strcat('./result/',method,'_out','.png'));
