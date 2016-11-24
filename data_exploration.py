#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:38:49 2016

@author: Qu
"""
#PassengerId -- A numerical id assigned to each passenger.
#Survived -- Whether the passenger survived (1), or didn't (0). We'll be making predictions for this column.
#Pclass -- The class the passenger was in -- first class (1), second class (2), or third class (3).
#Name -- the name of the passenger.
#Sex -- The gender of the passenger -- male or female.
#Age -- The age of the passenger. Fractional.
#SibSp -- The number of siblings and spouses the passenger had on board.
#Parch -- The number of parents and children the passenger had on board.
#Ticket -- The ticket number of the passenger.
#Fare -- How much the passenger paid for the ticker.
#Cabin -- Which cabin the passenger was in.
#Embarked -- Where the passenger boarded the Titanic.
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 


train = pd.read_csv('/Users/Qu/Downloads/train.csv')
test = pd.read_csv('/Users/Qu/Downloads/test.csv')

train.head()
train.tail()
test.head()

train.describe()
test.describe()

train.isnull().sum()
test.isnull().sum()
#可视化分析
plt.rc('font',size = 13)
fig = plt.figure(figsize = (18,8))
alpha = 0.6
#年龄分布
ax1 = plt.subplot2grid((2,4),(0,0))
train.Age.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
test.Age.value_counts().plot(kind='kde', label='test', alpha=alpha)
ax1.set_xlabel('Age')
ax1.set_title("What's the distribution of age?" )
plt.legend(loc='best')

ax2 = plt.subplot2grid((2,4),(0,1))
train.Pclass.value_counts().plot(kind = 'barh',color = 'r',label='train',alpha = alpha)
test.Pclass.value_counts().plot(kind = 'barh',color = 'b',label = 'test',alpha=alpha)
ax2.set_xlabel('Amount')
ax2.set_ylabel('Pcalss')
ax2.set_title('what is the distribution of Pcalss')
plt.legend(loc='best')

ax3 = plt.subplot2grid((2,4),(0,2))
train.SibSp.value_counts().plot(kind ='bar',label = 'train',color ='r',alpha= alpha)
test.SibSp.value_counts().plot(kind ='bar',label ='test',color ='b',alpha = alpha )
ax3.set_xlabel('SibSp')
ax3.set_title('what the distribution of SibSp')
plt.legend(loc ='best')

ax4 = plt.subplot2grid((2,4),(0,3))
train.Embarked.value_counts().plot(kind = 'bar',color ='r',label = 'train',alpha = alpha)
test.Embarked.value_counts().plot(kind = 'bar',color ='b',label = 'test',alpha = alpha)
ax4.set_xlabel('Embarked')
ax4.set_title('what the distribution of the Embarked')
plt.legend(loc ='best')

ax5 = plt.subplot2grid((2,4),(1,0),colspan = 2)
train.Fare.value_counts().plot(kind = 'kde',color ='r',label = 'train',alpha = alpha)
test.Fare.value_counts().plot(kind = 'kde',color = 'b',label = 'test',alpha=alpha)
ax5.set_xlabel('Fare')
ax5.set_title('what the distribution of Fare')
plt.legend(loc ='best')

ax6 = plt.subplot2grid((2,4),(1,2))
train.Parch.value_counts().plot(kind ='barh',color = 'r',label= 'train',alpha =alpha)
test.Parch.value_counts().plot(kind = 'barh',color ='b',label ='test',alpha = alpha)
ax6.set_xlabel('amount')
ax6.set_ylabel('Parch')
ax6.set_title('what the distribution of the Parch')
plt.legend(loc ='best')

ax7 = plt.subplot2grid((2,4),(1,3))
train.Sex.value_counts().plot(kind ='barh',color = 'r',label= 'train',alpha =alpha)
test.Sex.value_counts().plot(kind = 'barh',color ='b',label ='test',alpha = alpha)
ax7.set_xlabel('amount')
ax7.set_ylabel('Sex')
ax7.set_title('what the distribution of the Sex')
plt.legend(loc ='best')
plt.tight_layout()#自动调整图片合适大小

#训练集与测试集无太大区别，不需要处理即可进行下一步工作