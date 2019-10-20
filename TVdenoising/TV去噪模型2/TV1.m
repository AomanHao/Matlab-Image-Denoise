function [PSNRR,ImMAE,Is]=TV1(Io,In,tau,iter,lamda)
In=double(In);
Is=In;
t=0;
for i=1:iter
  i
  tic
%lamda=max(sum(sum(div(Is)*(Is-In)))./(xgm^2*255*255),0)
Is=Is+tau.*div(Is)-tau.*lamda*(Is-In);
figure(90)
imshow(Is,[]);
toc
t=t+toc;
   PSNRR(i)=psnr(Is,Io);
   NowPSNR=PSNRR(i)
   NowSNR=snr(Is,Io)
   NowMAE=mae(Is,Io)
   ImMAE(i)=NowMAE;
end
end
