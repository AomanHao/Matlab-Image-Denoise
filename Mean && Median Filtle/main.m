clear
close all
clc
%% ��ȡͼ��
I=imread('3096.jpg');
%% ��ɫͼ��ת�Ҷ�ͼ��
if size(I,3) == 3
   I=rgb2gray(I);
else
end
figure;imshow(I);title('(a)ԭʼͼ��');imwrite(I,'1.jpg');

%% ͼ���������
 I=imnoise(I,'salt & pepper',0.05); %�ӽ�������
% I=imnoise(I,'gaussian',0,0.01); % �Ӹ�˹����
figure;imshow(I);title('(b)����ͼ��');imwrite(I,'2.jpg');

[m,n]=size(I);
I=im2double(I);
r=3;%���򴰴�С
%% ����ʱ��ͳ��
t=cputime;
tic;
%% ��ֵ��ֵ�˲�
[I_mean,I_median]=compute_mean_median(I,r);
toc;
time_fcm_spatial_mean=cputime-t;%����ʱ��

figure;imshow(I_mean);title('��ֵ�˲�ͼ��');imwrite(I,'2.jpg');
figure;imshow(I_median);title('��ֵ�˲�ͼ��');imwrite(I,'3.jpg');

