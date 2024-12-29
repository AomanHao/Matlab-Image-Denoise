%% �������
% ���˲��� www.aomanhao.top
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
figure;imshow(I);title('(a)ԭʼͼ��');


% I=I;%��������
% I=imnoise(I,'speckle',deta_2);
% I=imnoise(I,'salt & pepper',0.05); %����ͼ
I=imnoise(I,'gaussian',0,0.001); % �Ӹ�˹����
imwrite(I,strcat('./result/','re','.png'));

%% ���ò���
w     = 1;       % bilateral filter half-width

tic;
method = 'bfilter2';
switch method
    case 'bfilter2'
        %% ˫���˲�
        sigma = [2 0.1]; % bilateral filter standard deviations
        result = bfilter2(I,w,sigma);
        
    case 'Joint_bfilter2'
        %% ����˫���˲�
        sigma = [2, 0.1, 0.5, 0.5]; 
        conf.BF_type = 'BiF';%BiF  SiF
        result = fun_Joint_bfilter2(I,w,sigma);
        
end
toc;
t=toc;
imwrite(double(result),strcat('./result/',method,'_out','.png'));
