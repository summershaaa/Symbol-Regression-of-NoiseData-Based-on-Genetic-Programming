# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:46:16 2019

@author: WinJX
"""

import numpy as np
import sympy as sp
from itertools import product
#定义变量和保留精度
x,y,z = sp.symbols('x,y,z')
np.set_printoptions(precision=6,suppress=6)

#方程
fml_n1 = x**3+x**2+x                  #n1
fml_n2 = x**4+x**3+x**2+x             #n2
fml_n3 = x**5+x**4+x**3+x**2+x        #n3
fml_n4 = x**6+x**5+x**4+x**3+x**2+x   #n4
fml_n5 = sp.sin(x**2)*sp.cos(x)-1     #n5
fml_n6 = sp.sin(x)+sp.sin(x+x**2)     #n6
fml_n7 = sp.log(x+1)+sp.log(x**2+1)   #n7
fml_n8 = sp.sqrt(x)                   #n8
fml_n9 = sp.sin(x)+sp.sin(y**2)       #n9
fml_n10 = 2*sp.sin(x)*sp.cos(y)       #n10

fml_k4 = x**3*sp.exp(-1*x)*sp.cos(x)*sp.sin(x)*((sp.sin(x))**2*sp.cos(x)-1) #k4
fml_k5 = (30*x*z)/((x-10)*y**2)       #k5
fml_k9 = sp.asinh(x)                  #k9
fml_k10 = x**y                        #k10
fml_k11 = x*y+sp.sin((x-1)*(y-1))     #k11
fml_k12 = x**4-x**3+y**2/2-y          #k12
fml_k13 = 6*sp.sin(x)*sp.cos(y)       #k13
fml_k14 = 8/(2+x**2+y**2)             #k14
fml_k15 = x**3/5+y**3/2-y-x           #k15

#数据集输入定义
train_n1x = np.random.uniform(-1,1,20)      #n1
test_n1x = np.random.uniform(-1,1,20)

train_n2x = np.random.uniform(-1,1,51)      #n2
test_n2x = np.random.uniform(-1,1,51)

train_n3x = np.random.uniform(-1,1,20)      #n3
test_n3x = np.random.uniform(-1,1,20)

train_n4x = np.random.uniform(-1,1,20)      #n4
test_n4x = np.random.uniform(-1,1,20) 

train_n5x = np.random.uniform(-1,1,51)      #n5
test_n5x = np.random.uniform(-1,1,51)

train_n6x = np.random.uniform(-1,1,20)      #n6 
test_n6x = np.random.uniform(-1,1,20)

train_n7x = np.random.uniform(0,2,20)       #n7
test_n7x = np.random.uniform(0,2,20)

train_n8x = np.random.uniform(0,4,40)       #n8
test_n8x = np.random.uniform(0,4,40)

train_n9x = np.random.uniform(-1,1,100)     #n9
train_n9y = np.random.uniform(-1,1,100)
test_n9x = np.random.uniform(-1,1,100)
test_n9y = np.random.uniform(-1,1,100)

train_n10x = np.random.uniform(-1,1,100)    #n10
train_n10y = np.random.uniform(-1,1,100)    
test_n10x = np.random.uniform(-1,1,100)
test_n10y = np.random.uniform(-1,1,100)


train_k4x = np.arange(0,10,0.05)            #k4
test_k4x = np.arange(0.05,10.05,0.05) 

train_k5x = np.random.uniform(-1,1,1000)    #k5
train_k5y = np.random.uniform(-1,1,1000)
train_k5z = np.random.uniform(1,2,1000)
test_k5x = np.random.uniform(-1,1,10000)
test_k5y = np.random.uniform(-1,1,10000)
test_k5z = np.random.uniform(1,2,10000)

train_k9x = np.arange(1,100,1)              #k9
test_k9x = np.arange(1,100,0.1)           

train_k10x = np.random.uniform(0,1,100)     #k10
train_k10y = np.random.uniform(0,1,100)
test_k10xy = np.arange(0,1,0.1)

train_k11x = np.random.uniform(-3,3,20)     #k11
train_k11y = np.random.uniform(-3,3,20)
test_k11xy = np.arange(-3,3,0.01)
 
train_k12x = np.random.uniform(-3,3,20)     #k12
train_k12y = np.random.uniform(-3,3,20)
test_k12xy = np.arange(-3,3,0.01)

train_k13x = np.random.uniform(-3,3,20)     #k13
train_k13y = np.random.uniform(-3,3,20)
test_k13xy = np.arange(-3,3,0.01)

train_k14x = np.random.uniform(-3,3,20)     #k14
train_k14y = np.random.uniform(-3,3,20)
test_k14xy = np.arange(-3,3,0.01)

train_k15x = np.random.uniform(-3,3,20)     #k15
train_k15y = np.random.uniform(-3,3,20)
test_k15xy = np.arange(-3,3,0.01)


def train_test_data(num,formual,target,datatype,path):
    """
    num:输入数据
    formual:方程
    target:输入数据的变量数
    datatype:输入数据的类型
    path：保存路径
    """
    #一元变量
    if target == 1:
        #定义方程
        func = sp.lambdify(x,formual,'numpy')
        value = num
        #代入数据得到结果
        rst = func(num)
        
    #二元变量
    if target == 2:
        func = sp.lambdify((x,y),formual,'numpy')
        #数据类型为随机均匀分布
        if datatype == 'U':
            val1 = np.random.choice(num[0],size = num[0].shape[0])  #第一个变量的数据
            val2 = np.random.choice(num[1],size = num[1].shape[0])  #第二个变量的数据
            value = np.c_[val1,val2]    #拼接两个变量的数据
        #数据类型为等间隔
        if datatype == 'E':
            #两个变量进行笛卡尔积组合
            value = np.array([x for x in product(num,num)])         
        rst = func(value[:,0],value[:,1])
    #三元变量  
    if target == 3:
        func = sp.lambdify((x,y,z),formual,'numpy')
        val1 = np.random.choice(num[0],size = num[0].shape[0])  #第一个变量的数据
        val2 = np.random.choice(num[1],size = num[1].shape[0])  #第二个变量的数据
        val3 = np.random.choice(num[2],size = num[2].shape[0])  #第三个变量的数据
        value = np.c_[val1,val2,val3]   #拼接三个变量的数据
        rst = func(value[:,0],value[:,1],value[:,2])
    #拼接变量数据和结果 
    rst = np.c_[value,rst]
    #存储保存
    np.savetxt(path,rst,fmt = '%.6f',delimiter = ' ')




#train_test_data(train_n1x,fml_n1,1,'U',path = 'n1_train.txt')
#
#train_test_data(train_n2x,fml_n2,1,'U',path = 'n2_train.txt')
#
#train_test_data(train_n3x,fml_n3,1,'U',path = 'n3_train.txt')
#
#train_test_data(train_n4x,fml_n4,1,'U',path = 'n4_train.txt')
#
#train_test_data(train_n5x,fml_n5,1,'U',path = 'n5_train.txt')
#
#train_test_data(train_n6x,fml_n6,1,'U',path = 'n6_train.txt')
#
#train_test_data(train_n7x,fml_n7,1,'U',path = 'n7_train.txt')
#
#train_test_data(train_n8x,fml_n8,1,'U',path = 'n8_train.txt')
#
#train_test_data([train_n9x,train_n9y],fml_n9,2,'U',path = 'n9_train.txt')
#
#train_test_data([train_n10x,train_n10y],fml_n10,2,'U',path = 'n10_train.txt')
#
#train_test_data(test_n1x,fml_n1,1,'U',path = 'n1_test.txt')
#
#train_test_data(test_n2x,fml_n2,1,'U',path = 'n2_test.txt')
#
#train_test_data(test_n3x,fml_n3,1,'U',path = 'n3_test.txt')
#
#train_test_data(test_n4x,fml_n4,1,'U',path = 'n4_test.txt')
#
#train_test_data(test_n5x,fml_n5,1,'U',path = 'n5_test.txt')
#
#train_test_data(test_n6x,fml_n6,1,'U',path = 'n6_test.txt')
#
#train_test_data(test_n7x,fml_n7,1,'U',path = 'n7_test.txt')
#
#train_test_data(test_n8x,fml_n8,1,'U',path = 'n8_test.txt')
#
#train_test_data([test_n9x,test_n9y],fml_n9,2,'U',path = 'n9_test.txt')
#
#train_test_data([test_n10x,test_n10y],fml_n10,2,'U',path = 'n10_test.txt')
#
#train_test_data(train_k4x,fml_k4,1,'E',path = 'k4_train.txt')
#
#train_test_data(train_k9x,fml_k9,1,'E',path = 'k9_train.txt')
#
#train_test_data([train_k10x,train_k10y],fml_k10,2,'U',path = 'k10_train.txt')
#
#train_test_data([train_k11x,train_k11y],fml_k11,2,'U',path = 'k11_train.txt')
#
#train_test_data([train_k12x,train_k12y],fml_k12,2,'U',path = 'k12_train.txt')
#
#train_test_data([train_k13x,train_k13y],fml_k13,2,'U',path = 'k13_train.txt')
#
#train_test_data([train_k14x,train_k14y],fml_k14,2,'U',path = 'k14_train.txt')
#
#train_test_data([train_k15x,train_k15y],fml_k15,2,'U',path = 'k15_train.txt')
#
#train_test_data(test_k4x,fml_k4,1,'E',path = 'k4_test.txt')
#
#train_test_data(test_k9x,fml_k9,1,'E',path = 'k9_test.txt')
#
#train_test_data(test_k10xy,fml_k10,2,'E',path = 'k10_test.txt')
#
#train_test_data(test_k11xy,fml_k11,2,'E',path = 'k11_test.txt')
#
#train_test_data(test_k12xy,fml_k12,2,'E',path = 'k12_test.txt')
#
#train_test_data(test_k13xy,fml_k13,2,'E',path = 'k13_test.txt')
#
#train_test_data(test_k14xy,fml_k14,2,'E',path = 'k14_test.txt')
#
#train_test_data(test_k15xy,fml_k15,2,'E',path = 'k15_test.txt')
#
#train_test_data([train_k5x,train_k5y,train_k5z],fml_k5,3,'E',path = 'k5_train.txt')
#
#train_test_data([test_k5x,test_k5y,test_k5z],fml_k5,3,'E',path = 'k5_test.txt')
 