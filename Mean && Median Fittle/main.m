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
% I=imnoise(I,'salt & pepper',0.05); %����ͼ
% I=imnoise(I,'gaussian',0,0.01); % �Ӹ�˹����

figure;imshow(I);title('(b)����ͼ��');imwrite(I,'2.jpg');
[m,n]=size(I);

% I = I/255��
I=im2double(I);
[I_mean,I_median]=compute_mean_median(I,r);%��չ����
% I_median=double(I_median);


I4 = I(:);  %% ��ͼ��ҶȰ�������
X_spatial_mean = I_mean(:);
% X_spatial_median = I_median(:);

%% ------------------------ fcm_spatial_mean------------------------
fcm_spatial_mean_label=zeros(m*n,1);
t=cputime;
tic;

toc;
time_fcm_spatial_mean=cputime-t;

figure;imshow(I);title('(a)ԭʼͼ��');imwrite(I,'1.jpg');
