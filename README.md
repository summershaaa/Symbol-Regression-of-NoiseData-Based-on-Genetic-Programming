# Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data
使用遗传规划解决带噪声数据的符号回归问题
1.研究背景：传统Genetic Programming方法能够在数据量大和无错误的情况下拟合出很好的表达式，但在存在噪声的数据中拟合结果会偏向噪声数据，噪声越大拟合效果越差。为了解决噪声数据带来的问题，结合了RANSAC方法处理噪声的思想，提了出一种改进的GP方法，在不同噪声率和噪声水平下对振荡函数和非振荡函数进行拟合，并对比两种改进前后的拟合效果。
2.测试的两个函数公式：
![函数公式](https://github.com/summershaaa/Genetic-Programming-to-Solve-Symbol-Regression-with-Noisy-Data/blob/master/Image/%E5%87%BD%E6%95%B0.png)
其中U[-1,1,20]表示x在区间[-1,1]内均匀分布地随机采样20个点


