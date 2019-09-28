clc;clear;
N=8000;    %�źŵ���
M=128;     %�˲�������
for k=1:N         
    x(k)=sin(k*2*pi/N)+2;    %�����ź�x
end
noise=0.3*randn(1,N);        %������˹���������
s=noise+x;     %�ź��м���������Ϊ�����ź�
d=x;           %������Ӧ=ԭ�ź�
w1=zeros(M,1);
w2=zeros(M,1);  %�˲����ĳ�ͷȨϵ����ֵΪ0
y1=zeros(N,1);
y2=zeros(N,1); %�˲�����������
%LMS�㷨
u1(1)=0.00001;
u2(1)=0.0001;
umax=0.001;    %����LMS�㷨����(n��=����
% mui=zeros(1,N);
for i=1:N-M
    if i<500
        u2(i)=umax;
    else u2(i)=u2(1);
    end
    for n=1:M     
            y1(i)=y1(i)+w1(n,1)*s(n+i-1);
            y2(i)=y2(i)+w2(n,1)*s(n+i-1);  %���=�˲���Ȩϵ��*����
    end
    e1(i)=d(i)-y1(i);
    e2(i)=d(i)-y2(i);  %�������
    for n=1:M
        w1(n,1)=w1(n,1)+u1*e1(i)*s(n+i-1);
        w2(n,1)=w2(n,1)+u2(i)*e2(i)*s(n+i-1);  %У���˲���Ȩϵ��w
    end
end
out1=y1;
out2=y2;
figure(1);
subplot(1,3,1);        %�ֱ𻭳�ԭ�źţ����������Ļ���źţ��˲���������ź�
plot(x);
subplot(1,3,2);
plot(s);
subplot(1,3,3);
title('LMS����Ӧ�㷨�˲��� �˲�����ź�');
plot(out1,'r');hold on;
plot(out2,'g');






