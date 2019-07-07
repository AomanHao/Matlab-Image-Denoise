%计算二维直方图的子函数
function [I_mean,I_median]=compute_mean_median(I,r) 
[m,n]=size(I);
I_median = medfilt2(I, [r,r]);
h = ones(r,r)/(r*r);
I_mean = filter2(h,I);
half_w=floor(r/2);
%扩展图像的边缘(按照镜子的方式)
extended_image1=[I(:,half_w:-1:1),I,I(:,end:-1:end-half_w-1)];   %扩展列
extended_image=[extended_image1(half_w:-1:1,:);extended_image1;extended_image1(end:-1:end-half_w-1,:)];  %扩展行

for i=[1,m]
    for j=1:n
        temp=extended_image(i:i+r-1,j:j+r-1);
        I_mean(i,j)=mean(temp(:));
        temp_sort = sort(temp(:));
        I_median(i,j)=temp_sort(ceil(r*r/2));        
    end
end
for j=[1,n]
    for i=2:m-1
        temp=extended_image(i:i+r-1,j:j+r-1);
        I_mean(i,j)=mean(temp(:));
        temp_sort = sort(temp(:));
        I_median(i,j)=temp_sort(ceil(r*r/2));        
    end
end