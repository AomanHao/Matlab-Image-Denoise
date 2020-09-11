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
%% 读取图像
I=imread('3096.jpg');

%% 彩色图像导向滤波
I = double(I)./ 255;
p = I;
r = 16;
eps = 0.1^2;
q = zeros(size(I));
q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);
figure;imshow(q);title('彩色滤波图像');
%% 彩色图像转灰度图像,导向滤波
if size(I,3) == 3
   I_g=rgb2gray(I);
else
end
figure;imshow(I_g);title('灰度图像');
p_g = I_g;
q_g = zeros(size(I_g));
q_g = guidedfilter(I_g, p_g, r, eps);
figure;imshow(q_g);title('灰度滤波图像');