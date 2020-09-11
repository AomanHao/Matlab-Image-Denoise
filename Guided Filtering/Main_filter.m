%% ������� 
% ���˲��� www.aomanhao.top
% Github https://github.com/AomanHao
% CSDN https://blog.csdn.net/Aoman_Hao
%
% �ο����������ļ������ַhttp://kaiminghe.com/eccv10/
%--------------------------------------

clear
close all
clc
%% ��ȡͼ��
I=imread('3096.jpg');

%% ��ɫͼ�����˲�
I = double(I)./ 255;
p = I;
r = 16;
eps = 0.1^2;
q = zeros(size(I));
q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);
figure;imshow(q);title('��ɫ�˲�ͼ��');
%% ��ɫͼ��ת�Ҷ�ͼ��,�����˲�
if size(I,3) == 3
   I_g=rgb2gray(I);
else
end
figure;imshow(I_g);title('�Ҷ�ͼ��');
p_g = I_g;
q_g = zeros(size(I_g));
q_g = guidedfilter(I_g, p_g, r, eps);
figure;imshow(q_g);title('�Ҷ��˲�ͼ��');