#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// additional functions/////////////////////////////////////
//add Salt-and-pepper noise
void addNoiseSoltPepperMono(Mat& src, Mat& dest, double per){//per控制椒盐噪点的含量
    cv::RNG rng;//随机数产生器
#pragma omp parallel for
    for (int j = 0; j<src.rows; j++){
        uchar* s = src.ptr(j);
        uchar* d = dest.ptr(j);
        for (int i = 0; i<src.cols; i++){
            //产生0~1均匀分布的随机数为 a1，用于控制是否在该像素点添加椒盐噪点
            double a1 = rng.uniform((double)0, (double)1);
            //如果随机数a1大于给定的值per，该点为原值，否则为椒盐噪点
            if (a1>per)d[i] = s[i];
            else{
                //产生0~1均匀分布的随机数为 a2，用于控制该噪点为椒噪点还是盐噪点
                double a2 = rng.uniform((double)0, (double)1);
                if (a2>0.5)d[i] = 0;
                else d[i] = 255;
            }
        }
    }
}
//添加高斯噪声
void addNoiseMono(Mat& src, Mat& dest, double sigma){
    Mat s;
    src.convertTo(s, CV_16S);
    Mat n(s.size(), CV_16S);
    //randn()将n填充为高斯分布均值为0，标准差为sigma的随机数
    randn(n, 0, sigma);
    Mat temp = s + n;//将原图与高斯噪声叠加在一起输出
    temp.convertTo(dest, CV_8U);
    /*
     CV_8U   8位无符号整型(0-255)
     CV_8S   8位有符号整型(-128-127)
     CV_16U  16位无符号整型(0-65535)
     CV_16S  16位有符号整型(-32768-32767)
     CV_32S  32位有符号整型(-2147483648-2147483647)
     CV_32F  32为浮点型
     CV_64F  64位浮点型
     */
}

void addNoise(Mat&src, Mat& dest, double sigma, double sprate = 0.0){
    //如果是单通道图片，调用一次增噪函数即可，否则拆分通道分别增噪然后合并
    if (src.channels() == 1){
        addNoiseMono(src, dest, sigma);//先添加高斯噪点
        if (sprate != 0)addNoiseSoltPepperMono(dest, dest, sprate);//然后添加椒盐噪点
        return;
    }else{
        Mat s[3];
        Mat d[3];
        split(src, s);
        for (int i = 0; i<src.channels(); i++){
            addNoiseMono(s[i], d[i], sigma);
            if (sprate != 0)addNoiseSoltPepperMono(d[i], d[i], sprate);
        }
        cv::merge(d, 3,dest);
    }
}

//计算一个通道的PSNR值，计算公式参考wiki
//https://zh.wikipedia.org/wiki/峰值信噪比
//https://zhuanlan.zhihu.com/p/40746930
static double getPSNR(Mat& src, Mat& dest){
    int i, j;
    double sse, mse, psnr;
    sse = 0.0;
    for (j = 0; j<src.rows; j++){
        uchar* d = dest.ptr(j);
        uchar* s = src.ptr(j);
        for (i = 0; i<src.cols; i++){
            sse += ((d[i] - s[i])*(d[i] - s[i]));
        }
    }
    if (sse == 0.0){
        return 0;
    }else{
        mse = sse / (double)(src.cols*src.rows);
        psnr = 10.0*log10((255 * 255) / mse);
        return psnr;
    }
}
//计算图片的PSNR
/*
 对于单通道图片，直接求即可。
 对于多通道图片，通常有三种方法：
 1：计算rgb三通道每个通道的psnr值，再求平均
 2：计算rgb三通道每个通道的mse值，再平均，得到psnr
 3：将rgb转换成yuv颜色空间，仅仅计算y分量的psnr，这个jvt用的比较多
 其中方法2和3用的比较多，1不常用！
 */
double calcPSNR(Mat& src, Mat& dest){
    Mat ssrc;
    Mat ddest;
    if (src.channels() == 1){
        src.copyTo(ssrc);
        dest.copyTo(ddest);
    }else{
        cvtColor(src, ssrc, CV_BGR2YUV);
        cvtColor(dest, ddest, CV_BGR2YUV);
    }
    double sn = getPSNR(ssrc, ddest);
    return sn;
}
////////////////////////////////////////////////////////////////////////////////
//NL-means 算法的实现
void nonlocalMeansFilter(Mat& src, Mat& dest, int templeteWindowSize, int searchWindowSize, double h, double sigma = 0.0){
    //邻域的大小不能超过搜索域的大小
    if (templeteWindowSize>searchWindowSize){
        cout << "searchWindowSize should be larger than templeteWindowSize" << endl;
        return;
    }
    //如果dest为空，则用与src相同大小的同型空白矩阵填充
    if (dest.empty())dest = Mat::zeros(src.size(), src.type());
    
    const int tr = templeteWindowSize >> 1;//tr为邻域的中心位置
    const int sr = searchWindowSize >> 1;//sr为搜索域的中心位置
    const int bb = sr + tr;//需增加的边界宽度
    const int D = searchWindowSize*searchWindowSize;//搜索域中的元素个数
    const int H = D / 2 + 1;//搜索域中的中心点位置
    const double div = 1.0 / (double)D;//均匀分布时，搜索域中的每个点的权重大小
    const int tD = templeteWindowSize*templeteWindowSize;//邻域中的元素个数
    const double tdiv = 1.0 / (double)(tD);//均匀分布时，搜索域中的每个点的权重大小
    
    //create large size image for bounding box;
    //copyMakeBorder()的参考用法：https://blog.csdn.net/qianqing13579/article/details/42323397
    Mat im;
    copyMakeBorder(src, im, bb, bb, bb, bb, cv::BORDER_DEFAULT);
    
    //weight computation;
    vector<double> weight(256 * 256 * src.channels());
    double* w = &weight[0];
    const double gauss_sd = (sigma == 0.0) ? h : sigma;//高斯标准差
    double gauss_color_coeff = -(1.0 / (double)(src.channels())) * (1.0 / (h*h));//高斯颜色系数
    int emax=0;
    //w[i]保存方差，即邻域平均欧氏距离对应的高斯加权权重，供后面计算出欧式距离后调用
    for (int i = 0; i < 256 * 256 * src.channels(); i++){
        double v = std::exp(max(i - 2.0*gauss_sd*gauss_sd, 0.0)*gauss_color_coeff);
        w[i] = v;
        if (v<0.001){
            emax = i;
            break;
        }
    }
    for (int i = emax; i < 256 * 256 * src.channels(); i++)w[i] = 0.0;
    
    if (src.channels() == 3){
        //mat.step
        //http://blog.sina.com.cn/s/blog_15183f5750102wu94.html
        //https://www.cnblogs.com/wangguchangqing/p/4016179.html
        const int cstep = (int)im.step - templeteWindowSize * 3;
        const int csstep = (int)im.step - searchWindowSize * 3;
#pragma omp parallel for
        for (int j = 0; j<src.rows; j++){//j for rows
            uchar* d = dest.ptr(j);
            int* ww = new int[D];//D 为搜索域中的元素数量，ww用于记录搜索域每个点的邻域方差
            double* nw = new double[D];//根据方差大小高斯加权归一化后的权重
            for (int i = 0; i<src.cols; i++){//i for cols
                double tweight = 0.0;
                //search loop
                uchar* tprt = im.data + im.step*(sr + j) + 3 * (sr + i);
                uchar* sptr2 = im.data + im.step*j + 3 * i;
                for (int l = searchWindowSize, count = D - 1; l--;){
                    uchar* sptr = sptr2 + im.step*(l);
                    for (int k = searchWindowSize; k--;){
                        //templete loop
                        int e = 0;
                        uchar* t = tprt;
                        uchar* s = sptr + 3 * k;
                        for (int n = templeteWindowSize; n--;){
                            for (int m = templeteWindowSize; m--;){
                                // computing color L2 norm
                                e += (s[0] - t[0])*(s[0] - t[0]) + (s[1] - t[1])*(s[1] - t[1]) + (s[2] - t[2])*(s[2] - t[2]);//L2 norm
                                s += 3;
                                t += 3;
                            }
                            t += cstep;
                            s += cstep;
                        }
                        const int ediv = e*tdiv;
                        ww[count--] = ediv;
                        //get weighted Euclidean distance
                        tweight += w[ediv];
                    }
                }
                //weight normalization
                if (tweight == 0.0){
                    for (int z = 0; z<D; z++) nw[z] = 0;
                    nw[H] = 1;
                }else{
                    double itweight = 1.0 / (double)tweight;
                    for (int z = 0; z<D; z++) nw[z] = w[ww[z]] * itweight;
                }
                double r = 0.0, g = 0.0, b = 0.0;
                uchar* s = im.ptr(j + tr); s += 3 * (tr + i);
                for (int l = searchWindowSize, count = 0; l--;){
                    for (int k = searchWindowSize; k--;)
                    {
                        r += s[0] * nw[count];
                        g += s[1] * nw[count];
                        b += s[2] * nw[count++];
                        s += 3;
                    }
                    s += csstep;
                }
                d[0] = saturate_cast<uchar>(r);
                d[1] = saturate_cast<uchar>(g);
                d[2] = saturate_cast<uchar>(b);
                d += 3;
            }//i
            delete[] ww;
            delete[] nw;
        }//j
    }else if (src.channels() == 1){
        const int cstep = (int)im.step - templeteWindowSize;//在邻域比较时，从邻域的上一行末尾跳至下一行开头
        const int csstep = (int)im.step - searchWindowSize;//搜索域循环中，从搜索域的上一行末尾跳至下一行开头
#pragma omp parallel for
        //下面两层嵌套循环：遍历每个图片的像素点
        for (int j = 0; j<src.rows; j++){
            uchar* d = dest.ptr(j);
            int* ww = new int[D];//D 为搜索域中的元素数量，ww用于记录搜索域每个点的邻域方差
            double* nw = new double[D];//根据方差大小高斯加权归一化后的权重
            for (int i = 0; i<src.cols; i++){
                //下面两层嵌套循环：遍历像素点（i，j）的搜索域点
                double tweight = 0.0;
                uchar* tprt = im.data + im.step*(sr + j) + (sr + i);//sr 为搜索域中心距
                uchar* sptr2 = im.data + im.step*j + i;
                for (int l = searchWindowSize, count = D - 1; l--;){
                    uchar* sptr = sptr2 + im.step*(l);
                    for (int k = searchWindowSize; k--;){
                        //下面两层嵌套循环：对于每个像素点（i，j）的搜索域点的邻域进行比较
                        int e = 0;//累计方差
                        uchar* t = tprt;
                        uchar* s = sptr + k;
                        for (int n = templeteWindowSize; n--;){
                            for (int m = templeteWindowSize; m--;){
                                // computing color L2 norm
                                e += (*s - *t)*(*s - *t);
                                s++;
                                t++;
                            }
                            t += cstep;
                            s += cstep;
                        }
                        const int ediv = e*tdiv;//tdiv 搜索域均一分布权重大小
                        ww[count--] = ediv;
                        //get weighted Euclidean distance
                        tweight += w[ediv];
                    }
                }
                //weight normalization权重归一化
                if (tweight == 0.0){
                    for (int z = 0; z<D; z++) nw[z] = 0;
                    nw[H] = 1;
                }else{
                    double itweight = 1.0 / (double)tweight;
                    for (int z = 0; z<D; z++) nw[z] = w[ww[z]] * itweight;
                }
                double v = 0.0;
                uchar* s = im.ptr(j + tr); s += (tr + i);
                for (int l = searchWindowSize, count = 0; l--;){
                    for (int k = searchWindowSize; k--;){
                        v += *(s++)*nw[count++];
                    }
                    s += csstep;
                }
                *(d++) = saturate_cast<uchar>(v);
            }//i
            delete[] ww;
            delete[] nw;
        }//j
    }
}

int main(int argc, char** argv){
    //(1) Reading image and add noise(standart deviation = 15)
    const double noise_sigma = 20.0;
    Mat src = imread("/Users/linweichen/Desktop/lena.jpg", 1);
    Mat snoise;
    Mat dest;
    addNoise(src, snoise, noise_sigma);
    //imwrite("/Users/linweichen/Desktop/lena_noise.jpg",snoise);
    
    //(2) preview conventional method with PSNR
    //(2-1) RAW
    cout << "RAW: " << calcPSNR(src, snoise) << endl << endl;
    //imwrite("noise.png", snoise);
    
    //(2-2) Gaussian Filter (7x7) sigma = 5
    int64 pre = getTickCount();
    GaussianBlur(snoise, dest, Size(7, 7), 5);
    cout << "time: " << 1000.0*(getTickCount() - pre) / (getTickFrequency()) << " ms" << endl;
    cout << "gaussian: " << calcPSNR(src, dest) << endl << endl;
    //imwrite("gaussian.png", dest);
    imshow("gaussian", dest);
    
    //(2-3) median Filter (3x3)
    pre = getTickCount();
    medianBlur(snoise, dest, 3);
    cout << "time: " << 1000.0*(getTickCount() - pre) / (getTickFrequency()) << " ms" << endl;
    cout << "median: " << calcPSNR(src, dest) << endl << endl;
    //imwrite("median.png", dest);
    imshow("median", dest);
    
    //(2-4) Bilateral Filter (7x7) color sigma = 35, space sigma = 5
    pre = getTickCount();
    bilateralFilter(snoise, dest, 7, 35, 5);
    cout << "time: " << 1000.0*(getTickCount() - pre) / (getTickFrequency()) << " ms" << endl;
    cout << "bilateral: " << calcPSNR(src, dest) << endl << endl;
    //imwrite("bilateral.png", dest);
    imshow("bilateral", dest);
    
    //(3) analizing of performance of Nonlocal means filter
    pre = getTickCount();
    nonlocalMeansFilter(snoise, dest, 3,7 , noise_sigma, noise_sigma);
    cout << "time: " << 1000.0*(getTickCount() - pre) / (getTickFrequency()) << " ms" << endl;
    cout << "nonlocal: " << calcPSNR(src, dest) << endl << endl;
    //imwrite("nonlocal.png", dest);
    imshow("original", src);
    imshow("noise", snoise);
    imshow("Non-local Means Filter", dest);
    while(waitKey(0));
    return 0;
}


