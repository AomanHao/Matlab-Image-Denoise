%% ������� 
% ���˲��� www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
%--------------------------------------
clear
close all
clc
%% %%%%%%%%%%%%%%%ͼ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I=imread('3096.jpg');

if size(I,3) == 3
   I=rgb2gray(I);
else
end
I=im2double(I);
figure;imshow(I);title('(a)ԭʼͼ��');imwrite(I,'1.jpg');
% I=I;%��������
% I=imnoise(I,'speckle',deta_2);
% I=imnoise(I,'salt & pepper',0.05); %����ͼ
img1=imnoise(I,'gaussian',0,0.01); % �Ӹ�˹����

%% ���ò���
w     = 5;       % bilateral filter half-width
sigma = [3 0.1]; % bilateral filter standard deviations

%% ˫���˲�
bflt_img1 = bfilter2(img1,w,sigma);

%% ���ͼ��
figure;imshow(I); title('Input Image');
figure;imshow(img1); title('Noise Image');
figure;imshow(bflt_img1);title('Result of Bilateral Filtering');