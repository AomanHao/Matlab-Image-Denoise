%% ������� 
% �����ʵ��ѧͼ�����Ŷ�-�º�
% ���˲��� www.aomanhao.top
% Github https://github.com/AomanHao
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