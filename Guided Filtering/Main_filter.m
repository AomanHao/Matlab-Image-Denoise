%% 程序分享
% 个人博客 www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
%
% 何凯明作者论文及代码地址http://kaiminghe.com/eccv10/
%--------------------------------------

clear
close all
clc
addpath('.\methods\')
%% 读取图像
img=imread('.\data\3096.jpg');
img = im2double(img);
[m,n,z] = size(img);
if  z>1
    I = rgb2gray(img);
else
    I = img;
end
Filter_type = 'guidedfilter';%guidedfilter
savepath = './result/';
if ~exist(savepath,'var')
    mkdir(savepath)
end
%% param
p = I;
r = 4;
eps = 0.1^2;
result = zeros(size(I));

switch Filter_type
    case  'guidedfilter'
        %% 引导滤波
        result(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
        
    case 'Weightguidedfilter'
        %% 权重引导滤波 <Weighted Guided Image Filtering>
        result(:, :, 1) = Weightguidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
        
end

imwrite(double(result),strcat(savepath,'result_',Filter_type,'_',num2str(eps),'.png'));