%% ������� 
% ���˲��� www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
%--------------------------------------


clear
close all
clc
%% ��ȡͼ��
I=imread('3096.jpg');

if size(I,3) == 3
   I=rgb2gray(I);
else
end
I=im2double(I);
figure;imshow(I);title('(a)ԭʼͼ��')
%% �������
%I=imnoise(I,'speckle',deta_2);
% I=imnoise(I,'salt & pepper',0.05); %����ͼ
I_noise=imnoise(I,'gaussian',0,0.01); % �Ӹ�˹����s
figure;imshow(I_noise);title('(b)����ͼ��');

%% ��̬ѧ�˲�
se=3; % the parameter of structuing element used for morphological reconstruction
data=w_recons_CO(I_noise,strel('square',se));

figure;imshow(data);title('ȥ��ͼ��');

