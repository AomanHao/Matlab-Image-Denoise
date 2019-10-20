clear all
close all
 I=imread('pepper.bmp');
 %I=imread('Barbara500.bmp');

 I=rgb2gray(I);
  I=double(I);
 figure(1),imshow(I,[]);
std_n=20;
var_n=std_n^2;
 NI=randn(size(I))*std_n;
In=I+NI;
% 
   save('In');
%  load('In'); 
figure(2);imshow(In,[]);
tau=0.15;
iter=130;
xgm=30;
lamda=0.05;
[PSNRR,ImMAE,Is]=TV1(I,In,tau,iter,lamda);

figure(4),imshow(Is,[]);
figure(5),imshow(Is-In,[]);
%»­PSNRR&ImMAEÍ¼
figure(8);
x=1:iter;
plot(x,PSNRR);
title('PSNRR');
 figure(10);
x=1:iter;
plot(x,ImMAE);
title('ImMAE');
% figure(11)
% x=1:iter;
% plot(x,SNR);
% title('SNR');
figure(15)
mesh(Is);