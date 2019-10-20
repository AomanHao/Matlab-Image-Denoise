function I=div(Io)
ep=0.0001;
Io=double(Io);
[m,n]=size(Io);
T=ones(m,n);

backx=Io(:,[2:end end])-Io;
backy=Io([2:end end],:)-Io;

forwardx=Io-Io(:,[1 1:end-1]);
forwardy=Io-Io([1 1:end-1],:);

term1=backx./(((backx.^2+minmod(backy,forwardy).^2)+ep).^(0.5));
term2=backy./(((backy.^2+minmod(backx,forwardx).^2)+ep).^(0.5));

I1=term1-term1(:,[1 1:end-1]);
I2=term2-term2([1 1:end-1],:);

I=I1+I2;
end
