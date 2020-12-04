%% ������� 
% ���˲��� www.aomanhao.top
% Github https://github.com/AomanHao
% ά���˲�ͼ����
%--------------------------------------

clear
close all
clc
%% ****************************ά���˲��;�ֵ�˲��ıȽ�*********************
img=imread('3096.jpg');
if size(img,3) == 3
   I=rgb2gray(img);
else
end

I_noise=imnoise(I,'gaussian',0,0.01);
figure;imshow(I);title('�Ҷ�ͼ');
figure;imshow(I_noise);title('����ͼ');
% ά���˲�
Out_wiener = wiener2(I_noise,[3 3]);
% ��ֵ�˲�
Mean_temp = ones(3,3)/9;
Out_mean = imfilter(I_noise,Mean_temp);
 
figure;
imshow(Out_wiener);title('ά���˲����');
figure;
imshow(uint8(Out_mean),[]);title('��ֵ�˲����');


%% *************************ά���˲�Ӧ����ͼ����ǿ***************************
for i = [1 2 3 4 5]
    K = wiener2(I,[5,5]);
end
 
K = K + I;
 
figure;imshow(I),title('ԭʼͼ��');
figure;imshow(K),title('��ǿ���ͼ��');



