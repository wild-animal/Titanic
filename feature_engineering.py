#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:39:30 2016

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

import sys
sys.path.append('/Users/Qu/Documents/Kaggle/Titanic/')
#import data_cleaning
from data_cleaning import train,test

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

#train = pd.read_csv('/Users/Qu/Downloads/train.csv')
#test = pd.read_csv('/Users/Qu/Downloads/test.csv')

full = pd.concat([train,test],ignore_index= True)
#计算名字数量
import re 
names = full.Name.map(lambda x: len(re.split(' ',x)))
_ = full.set_value(full.index, 'Name_len', names)
#_ = full.set_value(full.index, 'Names', names)
del names

title = full.Name.map(lambda x: re.compile(', (.*?)\.').findall(x)[0])
title[title=='Mme'] = 'Mrs'
title[title.isin(['Ms','Mlle'])] = 'Miss'
title[title.isin(['Don', 'Jonkheer'])] = 'Sir'
title[title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
title[title.isin(['Capt', 'Col', 'Major', 'Dr', 'Officer', 'Rev'])] = 'Officer'
_ = full.set_value(full.index, 'Title', title)
del title

_ = full.set_value(full.Cabin.isnull(),'Cabin','U0')

import re 

deck = full[full.Cabin.notnull()].Cabin.map(lambda x:re.compile('([a-zA-Z]+)').search(x).group())
deck = pd.factorize(deck)[0]
_ = full.set_value(full.index,'Deck',deck)
del deck

#room = full.Cabin.map(lambda x:re.compile('([0-9]+)').search(x).group())
checker = re.compile('([0-9]+)')
def roomNum(x):
    num = checker.search(x)
    if num:
        return int(num.group()) + 1
    else:
        return 1 
rooms = full.Cabin.map(roomNum)
_ = full.set_value(full.index,'Rooms',rooms)

#room数据归一化
#full.Rooms = full.Rooms/full.Rooms.sum()

full['Group_num'] = full.SibSp + full.Parch + 1

full['Group_size'] = pd.Series('M',index = full.index)
_ = full.set_value(full.Group_num >4, 'Group_size','L')
_ = full.set_value(full.Group_num == 1, 'Group_size','S')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
full['NorFare'] = pd.Series(scaler.fit_transform(full.Fare.reshape(-1,1)).reshape(-1), index=full.index)





























