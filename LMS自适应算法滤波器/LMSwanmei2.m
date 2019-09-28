clc;clear;
N=8000;    %信号点数
M=128;     %滤波器阶数
for k=1:N         
    x(k)=sin(k*2*pi/N)+2;    %产生信号x
end
noise=0.3*randn(1,N);        %产生高斯随机白噪声
s=noise+x;     %信号中加入噪声作为输入信号
d=x;           %期望响应=原信号
w1=zeros(M,1);
w2=zeros(M,1);  %滤波器的抽头权系数初值为0
y1=zeros(N,1);
y2=zeros(N,1); %滤波后的输出数组
%LMS算法
u1(1)=0.00001;
u2(1)=0.0001;
umax=0.001;    %基本LMS算法：μ(n）=常数
% mui=zeros(1,N);
for i=1:N-M
    if i<500
        u2(i)=umax;
    else u2(i)=u2(1);
    end
    for n=1:M     
            y1(i)=y1(i)+w1(n,1)*s(n+i-1);
            y2(i)=y2(i)+w2(n,1)*s(n+i-1);  %输出=滤波器权系数*输入
    end
    e1(i)=d(i)-y1(i);
    e2(i)=d(i)-y2(i);  %估计误差
    for n=1:M
        w1(n,1)=w1(n,1)+u1*e1(i)*s(n+i-1);
        w2(n,1)=w2(n,1)+u2(i)*e2(i)*s(n+i-1);  %校正滤波器权系数w
    end
end
out1=y1;
out2=y2;
figure(1);
subplot(1,3,1);        %分别画出原信号，加入噪声的混合信号，滤波后输出的信号
plot(x);
subplot(1,3,2);
plot(s);
subplot(1,3,3);
title('LMS自适应算法滤波器 滤波后的信号');
plot(out1,'r');hold on;
plot(out2,'g');






