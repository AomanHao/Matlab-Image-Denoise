% example: detail enhancement
% figure 6 in our paper

close all;

I = double(imread('3096.jpg')) / 255;
p = I;
figure;imshow(I);
r = 16;
eps = 0.1^2;

q = zeros(size(I));

q(:, :, 1) = guidedfilter(I(:, :, 1), p(:, :, 1), r, eps);
q(:, :, 2) = guidedfilter(I(:, :, 2), p(:, :, 2), r, eps);
q(:, :, 3) = guidedfilter(I(:, :, 3), p(:, :, 3), r, eps);

figure;imshow(q);
figure;imshow(q,[]);
w = (I - q);
figure;imshow(w);

I_enh_q2 = (I - q) * 2 + q;
figure();imshow(I_enh_q2);

I_enh_q3 = (I - q) * 3 + q;
figure();imshow(I_enh_q3);

I_enh_q4 = (I - q) * 4 + q;
figure();imshow(I_enh_q4);

I_enh_q5 = (I - q) * 5 + q;
figure();imshow(I_enh_q5);

I_enh_i2 = (I - q) * 2 + I;
figure();imshow(I_enh_i2);

I_enh_i3 = (I - q) * 3 + I;
figure();imshow(I_enh_i3);

I_enh_i4 = (I - q) * 4 + I;
figure();imshow(I_enh_i4);

I_enh_i5 = (I - q) * 5 + I;
figure();imshow(I_enh_i5);
