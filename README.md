# Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data
## 使用遗传规划解决带噪声数据的符号回归问题

#### 0.实验环境：
     Python3 , gplearn , numpy , sympy , matplotlib , seaborn

#### 1.研究背景：
   传统Genetic Programming方法能够在数据量大和无错误的情况下拟合出很好的表达式，但在存在噪声的数据中拟合结果会偏向噪声数据，噪声越大拟合效果越差。<br>为了解决噪声数据带来的问题，结合了RANSAC方法处理噪声的思想，提了出一种改进的GP方法，在不同噪声率和噪声水平下对振荡函数和非振荡函数进行拟合，并对比两种改进前后的拟合效果。
 > **改进效果对比图**
 * 左上图是训练数据及目标函数，50%的数据存在噪声。
 * 左下图是常规GP拟合的结果，因为使用所有数据训练，所以拟合结果会受到噪声数据的影响。
 * 中上图是改进GP的第一次拟合结果，因初始训练数据是随机选择的，所以效果较差。
 * 右上图是改进GP的第二次拟合结果，选择离第一次拟合结果最近的前N个数据进行训练，发现拟合效果好了不少。
 * 中下图是改进GP的第三次拟合结果，选择离第二次拟合结果最近的前N个数据进行训练，发现拟合结果已经非常接近目标函数。
 * 右下图是改进GP的第四次拟合结果，选择离第三次拟合结果最近的前N个数据进行训练，发现拟合结果和目标函数一致。
 
![分解改进过程](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E5%88%86%E8%A7%A3%E8%BF%87%E7%A8%8B.png)

#### 2.测试函数：

![函数公式](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E5%87%BD%E6%95%B0.png)

其中U[-1,1,20]表示x在区间[-1,1]内均匀分布地随机采样20个点。

#### 3.实验结果对比：
![第一个函数的两种方法拟合效果对比](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E5%AF%B9%E6%AF%94%E5%9B%BE1.png)
![第二个函数的两种方法拟合效果对比](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E5%AF%B9%E6%AF%94%E5%9B%BE2.png)

#### 4.可视化不同噪声率和不同噪声水平下的拟合优度和均方误差
![第一个函数的两种方法拟合对比](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/x%5E4%2Bx%5E3%2Bx%5E2%2Bx__%E5%8F%AF%E8%A7%86%E5%8C%96%E7%9B%B8%E5%85%B3%E6%80%A7%E5%92%8C%E8%AF%AF%E5%B7%AE.png)
![第二个函数的两种方法拟合对比](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/1%2Bsin(3x)__%E5%8F%AF%E8%A7%86%E5%8C%96%E7%9B%B8%E5%85%B3%E6%80%A7%E5%92%8C%E8%AF%AF%E5%B7%AE.png)


#### 5.对比表格

                                          函数1的实验数据对比表
                                                      
![第一个函数的两种方法拟合对比](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E6%95%B0%E6%8D%AE%E8%A1%A8%E6%A0%BC1.png)

                                          函数2的实验数据对比表
                                                      
![第二个函数的两种方法拟合对比](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E6%95%B0%E6%8D%AE%E8%A1%A8%E6%A0%BC2.png)

