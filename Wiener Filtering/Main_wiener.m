%% 程序分享 
% 个人博客 www.aomanhao.top
% Github https://github.com/AomanHao
% 维纳滤波图像处理
%--------------------------------------

clear
close all
clc
%% ****************************维纳滤波和均值滤波的比较*********************
img=imread('3096.jpg');
if size(img,3) == 3
   I=rgb2gray(img);
else
end

I_noise=imnoise(I,'gaussian',0,0.01);
figure;imshow(I);title('灰度图');
figure;imshow(I_noise);title('加噪图');
% 维纳滤波
Out_wiener = wiener2(I_noise,[3 3]);
% 均值滤波
Mean_temp = ones(3,3)/9;
Out_mean = imfilter(I_noise,Mean_temp);
 
figure;
imshow(Out_wiener);title('维纳滤波输出');
figure;
imshow(uint8(Out_mean),[]);title('均值滤波输出');


%% *************************维纳滤波应用于图像增强***************************
for i = [1 2 3 4 5]
    K = wiener2(I,[5,5]);
end
 
K = K + I;
 
figure;imshow(I),title('原始图像');
figure;imshow(K),title('增强后的图像');



