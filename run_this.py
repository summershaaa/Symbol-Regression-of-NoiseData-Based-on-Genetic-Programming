# -*- coding: utf-8 -*-
"""
Created on Fri May 10 17:18:41 2019

@author: WinJX
"""

import numpy as np
from sklearn.metrics import mean_squared_error
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#中文和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#禁用科学表示法和设置精度
np.set_printoptions(precision=6,suppress=True)


#添加噪声
def add_noise(data,noise_rate,noise_level):
    """
    data:需要添加噪声的数据
    noise_rate:噪声率，0-1之间
    noise_rate:噪声水平，(mu,sigma)
    先得出需要添加噪声的数据量和噪声大小，然后再随机添加到原数据中。
    比如原数据有20个数据点，需要添加6个噪声点，
    则可以构造一个带部分噪声的数据（14个零和6个噪声），
    打乱噪声数据顺序后再与原数据相加
    """
    #例如 ：g = [(0,0.5),(0,1),(0,2),(0,3)]
    #噪声水平
    mu,sigma = noise_level
    #噪声数据数量
    count = int(data.shape[0]*noise_rate)
    #非噪声数据数量
    pure = data.shape[0] - count
    #定义随机高斯噪声
    noise = rng.normal(mu,sigma,size = count)    
    zero = np.zeros(pure)
    noise_data = np.concatenate((noise,zero))
    #将噪声数据打乱
    np.random.shuffle(noise_data)
    #返回加上噪声后的数据
    return data+noise_data


#传统GP
def train(x,y_truth,X_train,y_train,X_test,y_test,target_func,noise_rate,noise_level):
    """
    x:  目标函数的分布范围
    y_truth: 目标函数的真实值
    X_train: 训练数据
    y_train: 训练数据值(带噪声)
    X_test: 测试数据
    y_test: 测试数据值
    noise_rate: 噪声率
    noise_level: 噪声水平
    得出用所有数据进行训练的拟合结果。拟合效果有可能会受噪声数据的影响
    """
    #查看训练所用的数据
    print('---训练数据---')
    print(np.c_[X_train,y_train]) 
    #定义符号回归器
    est_gp = SymbolicRegressor(population_size=5000,
                           function_set=['add','sub','mul','div'],#'sin','sqrt','cos'],#,'cos','sqrt','log','abs','neg','inv','tan'],
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,metric='mean absolute error',
                           parsimony_coefficient=0.01, random_state=0,const_range=(-1,1))
    #用训练集进行拟合训练   
    est_gp.fit(X_train.reshape(-1,1), y_train)
    #得到测试数据的预测值
    y_pred = est_gp.predict(X_test.reshape(-1,1))
    #得到R^2值
    score_gp = est_gp.score(X_test.reshape(-1,1), y_test)
    #训练集的均方误差
    test_mse = mean_squared_error(y_test,y_pred)
    print('拟合结果',str(est_gp._program))
    print('R^2 : %.6f'%score_gp)
    print('MSE : %.6f'%test_mse)
    
    #可视化目标曲线
    plt.xlabel('$x$',fontsize = 18)
    plt.ylabel('$y$',fontsize = 18)
    plt.plot(x,y_truth,label = target_func)
    plt.legend(loc = 'best',fontsize = 18)
    
    #可视化训练数据集
    plt.scatter(X_train,y_train,label = 'NoisyData',alpha = 0.9)
    plt.legend(loc = 'best',fontsize = 18)
        
    #可视化拟合曲线
    data = np.c_[X_test,y_pred]
    data = data[np.lexsort(data[:,::-1].T)]
    plt.plot(data[:,0], data[:,1], label = 'GP : '+str(est_gp._program))
    
    #标题
    fmt = '$R^2 =\/ {0:.6f}$ , $MSE =\/ {1:.6f}$'.format(score_gp,test_mse)
    plt.title(fmt,fontproperties = 'SimHei',fontsize = 20)   
    plt.legend(loc = 'best',fontsize = 18)



#改进GP
def ransac(x,y_truth,X_train,y_train,X_test,y_test,target_func,noise_rate,noise_level):
    """
    x:  目标函数的分布范围
    y_truth: 目标函数的真实值
    X_train: 训练数据
    y_train: 训练数据值(带噪声)
    X_test: 测试数据
    target_func :目标函数表达式
    y_test: 测试数据值
    noise_rate: 噪声率
    noise_level: 噪声水平
    利用部分数据集进行训练，并根据拟合结果来重新选择训练数据，
    通过这种方式来提高对噪声的鲁棒性。    
    """
    
    #最大迭代次数
    max_iter = 5
    #数据集大小
    length = X_train.shape[0]
    #噪声数据
    y_noise = y_train 
    #噪声数据数量
    noise_count = int(length*noise_rate)
    #训练数据数量，如果噪声率λ小于0.5，则选取(1-λ)length个数据点，
    #否则选取0.5length个数据点
    if noise_rate <= 0.5:
        pure_count = length - noise_count
    else:
        pure_count = length//2
    #迭代计数
    count = 0
    #测试集R^2得分
    test_score = []
    #测试集均方误差
    test_mse = []
    #拟合曲线表达式
    result = []
    #训练集
    train_data = np.c_[X_train,y_noise]
    print('------所有训练数据------')
    print(train_data)
    print('-----------------------')
    
    #随机采样初始训练数据集,定义一个顺序列表，将列表打乱取前qure_count个索引数据
    lst = list(range(length))
    np.random.shuffle(lst)
    #初始训练数据：数量为无噪声数据个数
    random_train_data = train_data[lst[:pure_count]]
    #保存每轮训练的数据集
    data_list = [0]*max_iter
    
    while count < max_iter:
        #保存当前训练数据集
        data_list[count] = random_train_data

        print('------------------------------第'+str(count)+'轮训练------------------------------')
        print('----该轮训练使用的数据----')
        print(random_train_data)
        print('-------------------------')
        #符号回归器
        est_gp = SymbolicRegressor(population_size=5000,
                           function_set=['add','sub','mul','div'],#'sin','sqrt','cos'],#,'cos','sqrt','log','abs','neg','inv','tan'],
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,metric='mean absolute error',
                           parsimony_coefficient=0.01, random_state=0,const_range=(-1,1))
        
        #用训练数据去拟合
        est_gp.fit(random_train_data[:,0].reshape(-1,1), random_train_data[:,1])
        #得到拟合表达式
        print('拟合结果 : ',est_gp._program)
        result.append(str(est_gp._program))
        #所有数据集的预测值
        y_pred = est_gp.predict(X_train.reshape(-1,1))
        #测试集的预测值
        ytest_pred = est_gp.predict(X_test.reshape(-1,1))
        #用于训练数据的预测值
        #ytrain_pred = est_gp.predict(random_train_data[:,0].reshape(-1,1))
        #测试集的R^2值
        score = est_gp.score(X_test.reshape(-1,1),y_test)
        test_score.append(score)
        #训练数据的均方误差
        mse = mean_squared_error(ytest_pred,y_test)
        test_mse.append(mse)
        
        #所有训练数据值与预测值的差值
        diff = abs(y_pred - y_train)
        #选取差值最小的前pure_count组数据，当做下一轮训练数据
        flag = np.where(diff < sorted(diff)[pure_count],1,0)
        temp_data = train_data[flag == 1]
        #如果新训练集和上轮数据一样，则退出循环
        if (temp_data == random_train_data).all():
            break
        else:
        #新的训练数据
            random_train_data = train_data[flag == 1]
            
        count += 1
        print('MSE : {0} , R^2 : {1} '.format(test_mse[-1],test_score[-1])) 
        
    ytest_pred = est_gp.predict(X_test.reshape(-1,1))
    ytest_score = est_gp.score(X_test.reshape(-1,1),y_test)
    ytest_mse = mean_squared_error(ytest_pred,y_test)
    
    #可视化目标函数
    plt.xlabel('$x$',fontsize = 18)
    plt.ylabel('$y$',fontsize = 18)
    plt.plot(x,y_truth,label = target_func)
    plt.legend(loc = 'best',fontsize = 18)
    
    #可视化训练数据
    plt.scatter(X_train,y_noise,label = 'NoisyData',alpha = 0.9)
    plt.legend(loc = 'best',fontsize = 18)
        
    #可视化拟合曲线 
    data = np.c_[X_test,ytest_pred]
    data = data[np.lexsort(data[:,::-1].T)]
    plt.plot(data[:,0], data[:,1], label = 'RCGP : '+str(est_gp._program))
    
    fmt = '$R^2 =\/ {0:.6f}$ , $MSE =\/ {1:.6f}$'.format(ytest_score,ytest_mse)
    plt.title(fmt,fontproperties = 'SimHei',fontsize = 20)
    plt.legend(loc = 'best',fontsize = 16)
    print(result)
    print('mse: ',test_mse)
    print('R^2: ',test_score)
    print()
    print("R^2 : %.6f"%test_score[-1])
    print("MSE : %.6f"%test_mse[-1])
    #print(est_gp.score(X_test.reshape(-1,1),y_test))
    return data_list
        
        


if __name__ == '__main__':

    rng = np.random.RandomState(0)
    
    #目标方程式
    x = np.linspace(-1,1,20)
    y_truth = x**4+x**3+x**2+x
    target = 'Objective_Function : $x^4+x^3+x^2+x$'
    
    #数据集接口
    train_data = np.loadtxt('./n2_train.txt')
    test_data = np.loadtxt('./n2_test.txt')
    X_train = train_data[:,0]
    y_train = train_data[:,1]
    X_test = test_data[:,0]
    y_test = test_data[:,1]
    
    #添加噪声，只需修改参数即可
    noise_rate,noise_level = 0.4,(0,2)
    y_train = add_noise(y_train,noise_rate,(noise_level[0],noise_level[-1]))
    
    #保存噪声数据，命名形式:0.4_g(0,2).txt
    np.savetxt('./{}_g({},{}).txt'.format(noise_rate,noise_level[0],noise_level[-1]),
               np.c_[X_train,y_train],fmt = '%.6f',delimiter = ' ')
    
    #可视化
    plt.figure(figsize=(24, 12))
    fmt = '$NoiseRate = {0}$ , $NoiseLevel = g({1},{2})$'.format(noise_rate,noise_level[0],noise_level[-1])
    plt.suptitle(fmt,fontproperties = 'SimHei',fontsize = 26)
    
    print('---------------------------------------GP---------------------------------------')
    plt.subplot(121)
    train(x,y_truth,X_train,y_train,X_test,y_test,target,noise_rate,(noise_level[0],noise_level[-1]))
   
    print('--------------------------------------RCGP--------------------------------------')
    plt.subplot(122)
    ransac(x,y_truth,X_train,y_train,X_test,y_test,target,noise_rate,(noise_level[0],noise_level[-1]))
    
    #保存图片，命名形式:0.4_g(0,2).png
    plt.savefig('./{}_g({},{}).png'.format(noise_rate,noise_level[0],noise_level[-1]))
