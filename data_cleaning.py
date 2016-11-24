#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:23:34 2016

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

#对embarked缺失值处理
train[train.Embarked.isnull()]

fig = plt.figure(figsize =(8,5))

ax = fig.add_subplot(1,1,1)
ax = train.boxplot(column='Fare',by = ['Embarked','Pclass'],ax=ax)
ax.set_ylim(0,200)
plt.axhline(y = 80,color = 'g')

_ = train.set_value(train.Embarked.isnull(),'Embarked','C')

#test的Fare缺失值处理
test[test.Fare.isnull()]
     
fig = plt.figure(figsize = (8,5))

ax = fig.add_subplot(1,1,1)
ax = test[(test.Embarked == 'S')&(test.Pclass == 3)].Fare.hist(bins = 100)
ax.set_xlabel('Fare')
ax.set_ylabel('amounts')

test[(test.Embarked == 'S')&(test.Pclass == 3)].Fare.value_counts()

_  = test.set_value(test.Fare.isnull(),'Fare',8.05)
